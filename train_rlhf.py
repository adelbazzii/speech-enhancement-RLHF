import copy
import csv
import os
import random
import sys
import warnings

import soundfile as sf
import torch
from natsort import natsorted
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "cmgan", "src"))

from eval_rlhf import evaluate_mos
from models.cmgan.src.models.generator import TSCNet as CMGANGenerator
from models.cmgan.src.utils import power_compress, power_uncompress
from models.metricgan_plus.model import Generator as MetricGANGenerator
from rlhf.buffer import ExperienceBuffer
from rlhf.loss import (
    cmgan_mse_loss,
    combined_loss,
    compute_j_theta,
    gaussian_kl,
    metricgan_mse_loss,
    ppo_clip_loss,
)
from rlhf.nisqa import get_nisqa_score, load_nisqa

warnings.filterwarnings("ignore", category=UserWarning)


MODEL = "cmgan"  # "cmgan" or "metricgan"
CMGAN_CKPT = "checkpoints/sft/cmgan/ckpt"
METRICGAN_CKPT = "checkpoints/sft/metricgan/CSIG-GAN_trial1.pth"
NISQA_CKPT = "checkpoints/nisqa/nisqa_mos_only.tar"
DATA_DIR = "data/voicebank_demand"
SAVE_DIR = f"checkpoints/rlhf/{MODEL}"
METRICS_PATH = f"logs/rlhf/{MODEL}_metrics.csv"
METRICS_FIELDS = ["step", "reward", "ppo_loss", "mse_loss", "total_loss", "test_mos"]

if MODEL == "cmgan":
    BATCH_SIZE = 2
    ACCUM_STEPS = 32
elif MODEL == "metricgan":
    BATCH_SIZE = 8
    ACCUM_STEPS = 8

EPSILON = 0.01
ALPHA = 0.7
BETA = 0.0001
LAMBDA = 0.0
SIGMA = 0.01

LR = 1e-6
MAX_STEPS = 2000
PPO_EPOCHS = 2

NUM_WORKERS = 2
LOG_INTERVAL = 10
SAVE_EVERY = 50
EVAL_INTERVAL = 10


def log_metrics(row):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    write_header = not os.path.exists(METRICS_PATH)
    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cmgan(ckpt_path, device):
    generator = CMGANGenerator().to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt)
        print(f"Loaded CMGAN checkpoint from {ckpt_path}")
    else:
        print("No CMGAN checkpoint found. Starting training from scratch.")
    return generator


def load_metricgan(ckpt_path, device):
    generator = MetricGANGenerator().to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        print(f"Loaded MetricGAN+ checkpoint from {ckpt_path}")
    else:
        print("No MetricGAN+ checkpoint found. Starting training from scratch.")
    return generator


SPLIT_DIRS = {
    "train": {"clean": "clean_trainset_28spk_wav", "noisy": "noisy_trainset_28spk_wav"},
    "test": {"clean": "clean_testset_wav", "noisy": "noisy_testset_wav"},
}


class VoiceBankDemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", cut_len=16000 * 2):
        self.cut_len = cut_len
        dirs = SPLIT_DIRS[split]
        self.clean_dir = os.path.join(data_dir, dirs["clean"])
        self.noisy_dir = os.path.join(data_dir, dirs["noisy"])
        self.clean_wav_name = natsorted(os.listdir(self.clean_dir))

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_np, _ = sf.read(clean_file)
        noisy_np, _ = sf.read(noisy_file)
        clean_ds = torch.from_numpy(clean_np).float()
        noisy_ds = torch.from_numpy(noisy_np).float()
        length = len(clean_ds)

        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds = torch.cat(
                [clean_ds] * units + [clean_ds[: self.cut_len % length]]
            )
            noisy_ds = torch.cat(
                [noisy_ds] * units + [noisy_ds[: self.cut_len % length]]
            )
        else:
            start = random.randint(0, length - self.cut_len)
            clean_ds = clean_ds[start : start + self.cut_len]
            noisy_ds = noisy_ds[start : start + self.cut_len]

        return clean_ds, noisy_ds, length


def create_dataloader(data_dir, split, batch_size, num_workers):
    ds = VoiceBankDemandDataset(data_dir, split)
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

    window = torch.hann_window(n_fft).to(device)
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
    rl_generator.eval()
    optimizer = torch.optim.Adam(rl_generator.parameters(), lr=LR)

    # sft policy
    sft_generator = copy.deepcopy(rl_generator)
    sft_generator.eval()
    for p in sft_generator.parameters():
        p.requires_grad = False

    # reward model
    nisqa_model, nisqa_args = load_nisqa(ckpt_path=NISQA_CKPT, device=device)

    # data loader
    dataloader = create_dataloader(
        data_dir=DATA_DIR, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # experience buffer
    buffer = ExperienceBuffer()

    n_fft = 400
    hop = 100
    data_iter = iter(dataloader)

    for step in tqdm(range(1, MAX_STEPS + 1), desc="CMGAN RLHF"):
        rewards_acc, ppo_losses_acc, mse_losses_acc, total_losses_acc = [], [], [], []

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
                sft_spec = torch.complex(sft_spec[..., 0], sft_spec[..., 1])
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

            rewards_acc.append(reward.mean().item())

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
                ppo_losses_acc.append(l_ppo.item())
                mse_losses_acc.append(l_mse.item())
                total_losses_acc.append(loss.item() * ACCUM_STEPS)
                # accumulate gradients
                loss.backward()

            # clip gradients and update weights
            torch.nn.utils.clip_grad_norm_(rl_generator.parameters(), max_norm=1.0)
            optimizer.step()

        # clear buffer
        buffer.clear()

        # log training metrics
        log_row = {k: None for k in METRICS_FIELDS}
        log_row["step"] = step
        if step % LOG_INTERVAL == 0:
            log_row["reward"] = sum(rewards_acc) / len(rewards_acc)
            log_row["ppo_loss"] = sum(ppo_losses_acc) / len(ppo_losses_acc)
            log_row["mse_loss"] = sum(mse_losses_acc) / len(mse_losses_acc)
            log_row["total_loss"] = sum(total_losses_acc) / len(total_losses_acc)
            print(
                f"Step {step}/{MAX_STEPS} | reward={log_row['reward']:.4f} | PPO Loss: {log_row['ppo_loss']:.4f} | MSE Loss: {log_row['mse_loss']:.4f} | Total Loss: {log_row['total_loss']:.4f} "
            )

        # evaluate on test set
        if step % EVAL_INTERVAL == 0:
            test_mos = evaluate_mos(
                rl_generator,
                "cmgan",
                nisqa_model,
                nisqa_args,
                device,
                cut_len=16000 * 2,
            )
            log_row["test_mos"] = test_mos
            print(f"Step {step}/{MAX_STEPS} | Test NISQA MOS: {test_mos:.4f}")

        if step % LOG_INTERVAL == 0 or step % EVAL_INTERVAL == 0:
            log_metrics(log_row)

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


def train_metricgan_rlhf(device):
    from models.metricgan_plus.signal_processing import (
        get_spec_and_phase,
        transform_spec_to_wav,
    )
    from rlhf.policy import metricgan_forward, metricgan_recompute_log_prob

    # rl policy
    rl_generator = load_metricgan(ckpt_path=METRICGAN_CKPT, device=device)
    rl_generator.lstm.dropout = 0
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
    dataloader = create_dataloader(
        data_dir=DATA_DIR, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # experience buffer
    buffer = ExperienceBuffer()

    data_iter = iter(dataloader)

    for step in tqdm(range(1, MAX_STEPS + 1), desc="MetricGAN+ RLHF"):
        rewards_acc, ppo_losses_acc, mse_losses_acc, total_losses_acc = [], [], [], []

        # fill experience buffer with old policy computations
        for _ in range(ACCUM_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            clean = batch[0].to(device)
            noisy = batch[1].to(device)

            with torch.no_grad():
                noise_mag, noise_phase = get_spec_and_phase(noisy)
                clean_mag, _ = get_spec_and_phase(clean)

                # sft forward pass
                sft_mask = sft_generator(noise_mag).clamp(min=0.05)
                sft_enh_mag = sft_mask * noise_mag
                sft_audio = transform_spec_to_wav(
                    torch.expm1(sft_enh_mag), noise_phase, signal_length=clean.shape[-1]
                )

                # rl forward pass
                rl_audio, mask_mean, action, log_prob_old = metricgan_forward(
                    generator=rl_generator,
                    noise_mag=noise_mag,
                    noise_phase=noise_phase,
                    sigma=SIGMA,
                    add_noise=True,
                    signal_length=clean.shape[-1],
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
                kl = gaussian_kl(mean_rl=mask_mean, mean_sft=sft_mask, sigma=SIGMA)
                j_theta = compute_j_theta(reward=reward, kl_div=kl, beta=BETA)

            rewards_acc.append(reward.mean().item())

            buffer.add(
                {
                    "noise_mag": noise_mag.detach(),
                    "clean_mag": clean_mag.detach(),
                    "action": action.detach(),
                    "log_prob_old": log_prob_old.detach(),
                    "j_theta": j_theta.detach(),
                }
            )

        # ppo update
        for _ in range(PPO_EPOCHS):
            optimizer.zero_grad()

            for exp in buffer:
                log_prob_new, enh_mag = metricgan_recompute_log_prob(
                    generator=rl_generator,
                    noise_mag=exp["noise_mag"],
                    stored_action=exp["action"],
                    sigma=SIGMA,
                )

                l_ppo = ppo_clip_loss(
                    log_prob_new=log_prob_new,
                    log_prob_old=exp["log_prob_old"],
                    j_theta=exp["j_theta"],
                    eps=EPSILON,
                )

                l_mse = metricgan_mse_loss(
                    enhanced_mag=enh_mag,
                    clean_mag=exp["clean_mag"],
                )

                loss = (
                    combined_loss(ppo_loss=l_ppo, mse_loss=l_mse, lam=LAMBDA)
                    / ACCUM_STEPS
                )
                ppo_losses_acc.append(l_ppo.item())
                mse_losses_acc.append(l_mse.item())
                total_losses_acc.append(loss.item() * ACCUM_STEPS)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(rl_generator.parameters(), max_norm=1.0)
            optimizer.step()

        buffer.clear()

        log_row = {k: None for k in METRICS_FIELDS}
        log_row["step"] = step
        if step % LOG_INTERVAL == 0:
            log_row["reward"] = sum(rewards_acc) / len(rewards_acc)
            log_row["ppo_loss"] = sum(ppo_losses_acc) / len(ppo_losses_acc)
            log_row["mse_loss"] = sum(mse_losses_acc) / len(mse_losses_acc)
            log_row["total_loss"] = sum(total_losses_acc) / len(total_losses_acc)
            print(
                f"Step {step}/{MAX_STEPS} | reward={log_row['reward']:.4f} | PPO Loss: {log_row['ppo_loss']:.4f} | MSE Loss: {log_row['mse_loss']:.4f} | Total Loss: {log_row['total_loss']:.4f}"
            )

        # evaluate on test set
        if step % EVAL_INTERVAL == 0:
            test_mos = evaluate_mos(
                rl_generator,
                "metricgan",
                nisqa_model,
                nisqa_args,
                device,
                cut_len=16000 * 2,
            )
            log_row["test_mos"] = test_mos
            print(f"Step {step}/{MAX_STEPS} | Test NISQA MOS: {test_mos:.4f}")

        if step % LOG_INTERVAL == 0 or step % EVAL_INTERVAL == 0:
            log_metrics(log_row)

        if step % SAVE_EVERY == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            ckpt_path = os.path.join(SAVE_DIR, f"metricgan_rlhf_step_{step}.pth")
            torch.save(
                {
                    "step": step,
                    "generator": rl_generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    print("MetricGAN+ RLHF training complete")


def main():
    device = get_device()
    print(f"Using device: {device}")

    if MODEL == "cmgan":
        train_cmgan_rlhf(device=device)
    elif MODEL == "metricgan":
        train_metricgan_rlhf(device=device)


if __name__ == "__main__":
    main()
