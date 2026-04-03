import copy
import os
import sys

# CMGAN's internal imports (e.g. "from models.conformer import ...") assume the working
# directory is models/cmgan/src/. Adding it to sys.path makes those imports resolve
# correctly when train_rlhf.py is run from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "cmgan", "src"))

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.cmgan.src.data.dataloader import DemandDataset
from models.cmgan.src.models.generator import TSCNet as CMGANGenerator
from models.cmgan.src.utils import power_compress, power_uncompress
from models.metricgan_plus.model import Generator as MetricGANGenerator
from rlhf.loss import gaussian_log_prob, gaussian_kl, compute_j_theta, ppo_clip_loss, cmgan_mse_loss, combined_loss
from rlhf.nisqa import load_nisqa, get_nisqa_score


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Creates a CMGAN generator (TSCNet) and loads weights from a checkpoint file if one exists.
# If no checkpoint is found, starts from scratch (untrained weights).
def load_cmgan(ckpt_path, device):
    generator = CMGANGenerator().to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt)
    else:
        print("No CMGAN checkpoint found. Starting training from scratch.")

    return generator


#Creates a MetricGAN+ generator and loads weights from a checkpoint file if one exists.
def load_metricgan(ckpt_path, device):
    generator = MetricGANGenerator().to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt)
    else:
        print("Np MetricGAN+ checkpoint found. Starting training from scratch.")

    return generator

# Builds a PyTorch DataLoader using CMGAN's DemandDataset.
# Loads VoiceBank+DEMAND audio from data_dir/split, shuffles, and batches it.
# NOTE:  MetricGAN+ has its own data loader in models/metricgan_plus/dataloader.py when train_metricgan_rlhf() is implemented, it should use that one instead.
def cmgan_create_dataloader(data_dir, split, batch_size, num_workers):
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
    Mirrors CMGAN's forward_generator_step preprocessing.
    Returns noisy_spec ready for the generator, and clean real/imag components for MSE loss.
    """
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
    noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)
    clean = torch.transpose(clean * c, 0, 1)

    window = torch.hamming_window(n_fft).to(device)
    noisy_spec = torch.stft(noisy, n_fft, hop, window=window, onesided=True, return_complex=False)
    clean_spec = torch.stft(clean, n_fft, hop, window=window, onesided=True, return_complex=False)

    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    clean_spec = power_compress(clean_spec)
    clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
    clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

    return noisy_spec, clean_real, clean_imag, window


def cmgan_forward(sft_generator, rl_generator, noisy_spec, sigma=0.01):
    """
    Runs both SFT (frozen) and RL (trainable) generators on the noisy spectrogram.
    Adds Gaussian noise to the RL output to form the action (paper Eq. 1).
    Returns the predicted masks from both policies and the noisy RL action.
    """
    with torch.no_grad():
        sft_real, sft_imag = sft_generator(noisy_spec)
        sft_real = sft_real.permute(0, 1, 3, 2)
        sft_imag = sft_imag.permute(0, 1, 3, 2)

    rl_real, rl_imag = rl_generator(noisy_spec)
    rl_real = rl_real.permute(0, 1, 3, 2)
    rl_imag = rl_imag.permute(0, 1, 3, 2)

    rl_real_noisy = rl_real + torch.randn_like(rl_real) * sigma
    rl_imag_noisy = rl_imag + torch.randn_like(rl_imag) * sigma

    return sft_real, sft_imag, rl_real, rl_imag, rl_real_noisy, rl_imag_noisy


def cmgan_to_audio(sft_real, sft_imag, rl_real_noisy, rl_imag_noisy, n_fft, hop, window):
    """
    Converts spectrogram masks back to audio waveforms via power uncompression and iSTFT.
    Returns sft_audio and rl_audio for NISQA scoring.
    """
    sft_spec = power_uncompress(sft_real, sft_imag).squeeze(1)
    sft_audio = torch.istft(sft_spec, n_fft, hop, window=window, onesided=True)

    rl_spec = power_uncompress(rl_real_noisy, rl_imag_noisy).squeeze(1)
    rl_audio = torch.istft(rl_spec, n_fft, hop, window=window, onesided=True)

    return sft_audio, rl_audio


def cmgan_compute_loss(
    sft_real, sft_imag, rl_real, rl_imag, rl_real_noisy, rl_imag_noisy,
    clean_real, clean_imag, sft_audio, rl_audio,
    nisqa_model, nisqa_args, device,
):
    """
    Computes the full RLHF loss for one accumulation step:
      1. NISQA reward: r_mos = NISQA(rl) - NISQA(sft)  (paper Eq. 2)
      2. KL divergence between RL and SFT policies
      3. J_theta = reward - beta * KL  (paper Eq. 3)
      4. PPO clip loss  (paper Eq. 4)
      5. MSE reconstruction loss  (paper Eq. 6)
      6. Combined loss = PPO + lambda * MSE  (paper Eq. 5)
    Returns loss, ppo_loss, mse_loss, reward for logging.
    """
    # Reward (paper Eq. 2)
    rl_mos = get_nisqa_score(nisqa_model, nisqa_args, rl_audio, device)
    sft_mos = get_nisqa_score(nisqa_model, nisqa_args, sft_audio, device)
    reward = rl_mos - sft_mos

    # KL divergence
    kl_div = gaussian_kl(rl_real, sft_real, sigma=0.01) + gaussian_kl(rl_imag, sft_imag, sigma=0.01)

    # J_theta (paper Eq. 3)
    j_theta = compute_j_theta(reward, kl_div, beta=0.0001)

    # Log probabilities
    log_prob_new = (
        gaussian_log_prob(rl_real_noisy, rl_real, sigma=0.01)
        + gaussian_log_prob(rl_imag_noisy, rl_imag, sigma=0.01)
    )
    log_prob_old = (
        gaussian_log_prob(rl_real_noisy, sft_real, sigma=0.01)
        + gaussian_log_prob(rl_imag_noisy, sft_imag, sigma=0.01)
    )

    # Losses (paper Eq. 4, 5, 6)
    ppo_loss = ppo_clip_loss(log_prob_new, log_prob_old, j_theta, eps=0.01)
    mse_loss = cmgan_mse_loss(rl_real, rl_imag, clean_real, clean_imag, alpha=0.7)
    loss = combined_loss(ppo_loss, mse_loss, lam=1.0)

    return loss, ppo_loss, mse_loss, reward


def train_cmgan_rlhf(device, lr, data_dir, nisqa_ckpt_path, ckpt_path, batch_size, num_workers, max_steps, accum_steps):
    # RL policy (trainable) and SFT policy (frozen reference)
    rl_generator = load_cmgan(ckpt_path=ckpt_path, device=device)
    rl_generator.train()

    sft_generator = copy.deepcopy(rl_generator)
    sft_generator.eval()
    for p in sft_generator.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(rl_generator.parameters(), lr=lr)
    dataloader = cmgan_create_dataloader(data_dir=data_dir, split="train", batch_size=batch_size, num_workers=num_workers)
    nisqa_model, nisqa_args = load_nisqa(nisqa_ckpt_path, device)

    n_fft = 400  # matching CMGAN's original training setup
    hop = 100

    data_iter = iter(dataloader)
    for step in tqdm(range(1, max_steps + 1), desc="CMGAN RLHF"):
        optimizer.zero_grad()

        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            clean = batch[0].to(device)
            noisy = batch[1].to(device)

            noisy_spec, clean_real, clean_imag, window = cmgan_preprocess(noisy, clean, n_fft, hop, device)

            sft_real, sft_imag, rl_real, rl_imag, rl_real_noisy, rl_imag_noisy = cmgan_forward(sft_generator, rl_generator, noisy_spec)
            
            sft_audio, rl_audio = cmgan_to_audio(sft_real, sft_imag, rl_real_noisy, rl_imag_noisy, n_fft, hop, window)
            
            loss, ppo_loss, mse_loss, reward = cmgan_compute_loss(sft_real, sft_imag, rl_real, rl_imag, rl_real_noisy,
                                                                   rl_imag_noisy, clean_real, clean_imag, sft_audio, 
                                                                   rl_audio, nisqa_model, nisqa_args, device,)
            
            (loss / accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(rl_generator.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | loss={loss.item():.4f} | ppo={ppo_loss.item():.4f} | mse={mse_loss.item():.4f} | reward={reward.mean().item():.4f}")


def train_metrocgan_rlhf(device):
    rl_generator = load_metricgan(device=device)


def main():
    # --- Configuration ---
    model         = "cmgan"     # "cmgan" or "metricgan"
    ckpt_path     = "models/cmgan/src/best_ckpt/best_model"
    nisqa_ckpt    = "nisqa/weights/nisqa_mos_only.tar"
    data_dir      = "data/voicebank_demand"
    save_dir      = "checkpoints/rlhf"
    lr            = 1e-6   # from paper
    batch_size    = 4      # from paper
    accum_steps   = 16     # effective batch size = 4 * 16 = 64 (from paper)
    max_steps     = 1000
    num_workers   = 2

    device = get_device()
    print(f"Using device: {device}")
    # os.makedirs(save_dir, exist_ok=True)

    if model == "cmgan":
        train_cmgan_rlhf(
            device=device,
            lr=lr,
            data_dir=data_dir,
            nisqa_ckpt_path=nisqa_ckpt,
            ckpt_path=ckpt_path,
            batch_size=batch_size,
            num_workers=num_workers,
            max_steps=max_steps,
            accum_steps=accum_steps,
        )
    elif model == "metricgan":
        train_metrocgan_rlhf(device=device)


if __name__ == "__main__":
    main()
