import copy
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import third_party_path_setup

from models.cmgan.src.data.dataloader import DemandDataset
from models.cmgan.src.models.generator import TSCNet as CMGANGenerator
from models.cmgan.src.utils import power_compress, power_uncompress
from models.metricgan_plus.model import Generator as MetricGANGenerator

from rlhf.buffer import ExperienceBuffer
from rlhf.loss import (
    cmgan_mse_loss,
    combined_loss,
    compute_j_theta,
    gaussian_kl,
    ppo_clip_loss,
)
from rlhf.nisqa import get_nisqa_score, load_nisqa

MODEL = "cmgan"  # "cmgan" or "metricgan"
CMGAN_CKPT = "models/cmgan/src/best_ckpt/ckpt"
METRICGAN_CKPT = "models/metricgan-plus/checkpoints"
NISQA_CKPT = "nisqa/weights/nisqa_mos_only.tar"
DATA_DIR = "data/voicebank_demand"
SAVE_DIR = "checkpoints/rlhf"

if MODEL == "cmgan":
    BATCH_SIZE = 4
    ACCUM_STEPS = 16
elif MODEL == "metricgan":
    BATCH_SIZE = 8
    ACCUM_STEPS = 8

EPSILON = 0.01
ALPHA = 0.7
BETA = 0.0001
LAMBDA = 1.0
SIGMA = 0.01

LR = 1e-6
MAX_STEPS = 2000
PPO_EPOCHS = 5

NUM_WORKERS = 2
LOG_INTERVAL = 10
SAVE_EVERY = 50


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cmgan(ckpt_path, device):
    generator = CMGANGenerator(num_channel=64, num_features=201).to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt)
        print(f"Loaded CMGAN checkpoint from {ckpt_path}")
    else:
        print("No CMGAN checkpoint found. Starting training from scratch.")
    return generator


def load_metricgan(ckpt_path, device):
    generator = MetricGANGenerator(causal=False).to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        print(f"Loaded MetricGAN+ checkpoint from {ckpt_path}")
    else:
        print("No MetricGAN+ checkpoint found. Starting training from scratch.")
    return generator


def create_dataloader(data_dir, split, batch_size, num_workers):
    ds = DemandDataset(os.path.join(data_dir, split))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dl


def cmgan_preprocess(noisy, clean, n_fft, hop, device):
    """
    Normalizes audio and converts to power-compressed spectrograms.
    Returns noisy_spec ready for the generator, and clean real/imag components for MSE loss.
    """
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)
    clean = torch.transpose(clean * c, 0, 1)

    window = torch.hamming_window(n_fft).to(device)
    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=window, onesided=True, return_complex=False
    )
    clean_spec = torch.stft(
        clean, n_fft, hop, window=window, onesided=True, return_complex=False
    )

    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    clean_spec = power_compress(clean_spec)
    clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
    clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

    return noisy_spec, clean_real, clean_imag, window


def train_cmgan_rlhf(device):
    from rlhf.policy import cmgan_forward, cmgan_recompute_log_prob

    # rl policy
    rl_generator = load_cmgan(ckpt_path=CMGAN_CKPT, device=device)
    rl_generator.train()
    optimizer = torch.optim.Adam(rl_generator.parameters(), lr=LR)

    # sft policy
    sft_generator = copy.deepcopy(rl_generator)
    sft_generator.eval()
    for p in sft_generator.parameters():
        p.requires_grad = False

    # reward model
    nisqa_model, nisqa_args = load_nisqa(ckpt_path=NISQA_CKPT, device=device)

    # data loader
    """
    Expects the following directory structure:
    data/
        voicebank_demand/
            train/
                clean/
                noisy/
            test/
                clean/
                noisy
    """
    dataloader = create_dataloader(
        data_dir=DATA_DIR, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # experience buffer
    buffer = ExperienceBuffer()

    n_fft = 400
    hop = 100
    data_iter = iter(dataloader)

    for step in tqdm(range(1, MAX_STEPS + 1), desc="CMGAN RLHF"):
        # fill expereince buffer with old policy computations
        for _ in range(ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            clean = batch[0].to(device)
            noisy = batch[1].to(device)

            noisy_spec, clean_real, clean_imag, window = cmgan_preprocess(
                noisy=noisy, clean=clean, n_fft=n_fft, hop=hop, device=device
            )

            with torch.no_grad():
                # sft forward pass
                sft_real, sft_imag = sft_generator(noisy_spec)
                sft_real = sft_real.permute(0, 1, 3, 2)
                sft_imag = sft_imag.permute(0, 1, 3, 2)
                sft_spec = power_uncompress(sft_real, sft_imag).squeeze(1)
                sft_audio = torch.istft(
                    sft_spec, n_fft, hop, window=window, onesided=True
                )

                # rl forward pass
                rl_audio, action_mean, action, log_prob_old = cmgan_forward(
                    generator=rl_generator,
                    noisy_spec=noisy_spec,
                    window=window,
                    sigma=SIGMA,
                    add_noise=True,
                )

                # reward
                rl_mos = get_nisqa_score(
                    nisqa_model=nisqa_model,
                    model_args=nisqa_args,
                    audio_tensor=rl_audio,
                    device=device,
                )
                sft_mos = get_nisqa_score(
                    nisqa_model=nisqa_model,
                    model_args=nisqa_args,
                    audio_tensor=sft_audio,
                    device=device,
                )
                reward = rl_mos - sft_mos

                # kl and j_theta
                kl = gaussian_kl(
                    mean_rl=torch.cat([action_mean[0], action_mean[1]], dim=1),
                    mean_sft=torch.cat([sft_real, sft_imag], dim=1),
                    sigma=SIGMA,
                )
                j_theta = compute_j_theta(reward=reward, kl_div=kl, beta=BETA)

            buffer.add(
                {
                    "noisy_spec": noisy_spec.detach(),
                    "clean_real": clean_real.detach(),
                    "clean_imag": clean_imag.detach(),
                    "action": (action[0].detach(), action[1].detach()),
                    "log_prob_old": log_prob_old.detach(),
                    "j_theta": j_theta.detach(),
                }
            )

        # ppo update
        for _ in range(PPO_EPOCHS):
            # zero out gradients
            optimizer.zero_grad()

            # loop through batches in buffer and accumulate gradients
            for exp in buffer:
                # recompute log probability under current policy
                log_prob_new, (est_real, est_imag) = cmgan_recompute_log_prob(
                    generator=rl_generator,
                    noisy_spec=exp["noisy_spec"],
                    stored_action=exp["action"],
                    sigma=SIGMA,
                )

                # ppo loss
                l_ppo = ppo_clip_loss(
                    log_prob_new=log_prob_new,
                    log_prob_old=exp["log_prob_old"],
                    j_theta=exp["j_theta"],
                    eps=EPSILON,
                )

                # mse loss
                l_mse = cmgan_mse_loss(
                    est_real=est_real,
                    est_imag=est_imag,
                    clean_real=exp["clean_real"],
                    clean_imag=exp["clean_imag"],
                    alpha=ALPHA,
                )

                # combines loss
                loss = (
                    combined_loss(ppo_loss=l_ppo, mse_loss=l_mse, lam=LAMBDA)
                    / ACCUM_STEPS
                )
                # accumulate gradients
                loss.backward()

            # update weights
            optimizer.step()

        # clear buffer
        buffer.clear()

        # log training metrics
        if step % LOG_INTERVAL == 0:
            print(
                f"Step {step}/{MAX_STEPS} | reward={reward.mean().item():.4f} | PPO Loss: {l_ppo.item():.4f} | MSE Loss: {l_mse.item():.4f} | Total Loss: {loss.item():.4f} "
            )

        # save checkpoint
        if step % SAVE_EVERY == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            ckpt_path = os.path.join(SAVE_DIR, f"cmgan_rlhf_step_{step}.pth")
            torch.save(
                {
                    "step": step,
                    "generator": rl_generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print("CMGAN RLHF training complete")


def train_metrocgan_rlhf(device):
    rl_generator = load_metricgan(device=device)


def main():
    # --- Configuration ---
    model = "cmgan"  # "cmgan" or "metricgan"
    ckpt_path = "models/cmgan/src/best_ckpt/best_model"
    nisqa_ckpt = "nisqa/weights/nisqa_mos_only.tar"


    device = get_device()
    print(f"Using device: {device}")
    # os.makedirs(save_dir, exist_ok=True)

    if model == "cmgan":
        train_cmgan_rlhf(device=device)
    elif model == "metricgan":
        train_metrocgan_rlhf(device=device)


if __name__ == "__main__":
    main()
