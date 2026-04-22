"""Evaluate CMGAN or MetricGAN+ checkpoints on VoiceBank+DEMAND test set."""

import os
import sys
import warnings

import numpy as np
import soundfile as sf
import torch
from natsort import natsorted
from pesq import pesq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "cmgan", "src"))

from models.cmgan.src.models.generator import TSCNet as CMGANGenerator
from models.cmgan.src.utils import power_compress, power_uncompress
from models.metricgan_plus.metric_functions.metric_helper import SSNR
from models.metricgan_plus.model import Generator as MetricGANGenerator
from models.metricgan_plus.signal_processing import (
    get_spec_and_phase,
    transform_spec_to_wav,
)
from rlhf.nisqa import get_nisqa_score, load_nisqa

warnings.filterwarnings("ignore", category=UserWarning)

MODEL = "metricgan"  # "cmgan" or "metricgan"

CMGAN_CKPT = "checkpoints/rlhf/lambda=0/cmgan/cmgan_rlhf_step_2000.pth"
METRICGAN_CKPT = "checkpoints/rlhf/lambda=0/metricgan/metricgan_rlhf_step_2000.pth"

CKPT = CMGAN_CKPT if MODEL == "cmgan" else METRICGAN_CKPT

NISQA_CKPT = "checkpoints/nisqa/nisqa_mos_only.tar"
DATA_DIR = "data/voicebank_demand"
SR = 16000

SPLIT_DIRS = {
    "test": {"clean": "clean_testset_wav", "noisy": "noisy_testset_wav"},
}


def si_sdr(ref, est):
    """Scale-Invariant Signal-to-Distortion Ratio."""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    dot = np.sum(ref * est)
    s_target = dot * ref / (np.sum(ref**2) + 1e-10)
    e_noise = est - s_target
    return 10 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-10))


def load_generator(model_type, ckpt_path, device):
    """Load a generator checkpoint (auto-detects raw state_dict vs RLHF wrapper)."""
    if model_type == "cmgan":
        generator = CMGANGenerator().to(device)
    elif model_type == "metricgan":
        generator = MetricGANGenerator().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    generator.load_state_dict(state)

    generator.eval()
    return generator


@torch.no_grad()
def enhance_cmgan(generator, noisy_np, device):
    """Run CMGAN on a single utterance, return enhanced numpy array."""
    n_fft, hop = 400, 100
    window = torch.hann_window(n_fft).to(device)
    noisy = torch.from_numpy(noisy_np).float().unsqueeze(0).to(device)

    c = torch.sqrt(noisy.size(-1) / torch.sum(noisy**2.0, dim=-1))
    noisy = noisy * c

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=window, onesided=True, return_complex=False
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)

    est_real, est_imag = generator(noisy_spec)
    est_real = est_real.permute(0, 1, 3, 2)
    est_imag = est_imag.permute(0, 1, 3, 2)

    est_spec = power_uncompress(est_real, est_imag).squeeze(1)
    est_spec = torch.complex(est_spec[..., 0], est_spec[..., 1])
    enhanced = torch.istft(est_spec, n_fft, hop, window=window, onesided=True)
    return enhanced


@torch.no_grad()
def enhance_metricgan(generator, noisy_np, device):
    """Run MetricGAN+ on a single utterance, return enhanced tensor."""
    noisy = torch.from_numpy(noisy_np).float().unsqueeze(0).to(device)
    noise_mag, noise_phase = get_spec_and_phase(noisy)
    mask = generator(noise_mag).clamp(min=0.05)
    enh_mag = mask * noise_mag
    enhanced = transform_spec_to_wav(
        torch.expm1(enh_mag), noise_phase, signal_length=noisy.shape[-1]
    )
    return enhanced


def evaluate(
    generator,
    model_type,
    nisqa_model,
    nisqa_args,
    device,
    full_metrics=True,
    cut_len=None,
):
    """Evaluate a generator on the test set.

    Args:
        full_metrics: If True, compute all metrics (NISQA, PESQ, SSNR, SI-SDR).
                      If False, compute only NISQA MOS (faster, used during training).
        cut_len: If set, truncate each test clip to this many samples before
                 enhancement (used during training to cap memory). None = native length.

    Returns:
        dict of metric_name -> mean value
    """
    was_training = generator.training
    generator.eval()
    test_noisy_dir = os.path.join(DATA_DIR, SPLIT_DIRS["test"]["noisy"])
    test_clean_dir = os.path.join(DATA_DIR, SPLIT_DIRS["test"]["clean"])
    filenames = natsorted(os.listdir(test_noisy_dir))

    enhance_fn = enhance_cmgan if model_type == "cmgan" else enhance_metricgan

    all_mos, all_pesq, all_ssnr, all_sisdr = [], [], [], []

    for fname in filenames:
        if device == "cuda":
            torch.cuda.empty_cache()
        noisy_np, _ = sf.read(os.path.join(test_noisy_dir, fname))
        if cut_len is not None and len(noisy_np) > cut_len:
            noisy_np = noisy_np[:cut_len]

        enhanced = enhance_fn(generator, noisy_np, device)

        # NISQA MOS
        mos = get_nisqa_score(nisqa_model, nisqa_args, enhanced, device)
        all_mos.append(mos.item())

        if full_metrics:
            clean_np, _ = sf.read(os.path.join(test_clean_dir, fname))
            enh_np = enhanced.squeeze(0).cpu().numpy()
            min_len = min(len(clean_np), len(enh_np))
            clean_np = clean_np[:min_len]
            enh_np = enh_np[:min_len]

            all_pesq.append(pesq(SR, clean_np, enh_np, "wb"))
            overall_snr, _ = SSNR(clean_np, enh_np, srate=SR)
            all_ssnr.append(overall_snr)
            all_sisdr.append(si_sdr(clean_np, enh_np))

    if was_training:
        generator.train()

    results = {"NISQA MOS": np.mean(all_mos)}
    if full_metrics:
        results["PESQ"] = np.mean(all_pesq)
        results["SSNR"] = np.mean(all_ssnr)
        results["SI-SDR"] = np.mean(all_sisdr)

    return results


def evaluate_mos(generator, model_type, nisqa_model, nisqa_args, device, cut_len=None):
    """Quick evaluation returning only mean NISQA MOS (used during training)."""
    return evaluate(
        generator,
        model_type,
        nisqa_model,
        nisqa_args,
        device,
        full_metrics=False,
        cut_len=cut_len,
    )["NISQA MOS"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    generator = load_generator(MODEL, CKPT, device)
    nisqa_model, nisqa_args = load_nisqa(NISQA_CKPT, device)

    print(f"Evaluating {MODEL} checkpoint: {CKPT}")
    results = evaluate(
        generator, MODEL, nisqa_model, nisqa_args, device,
        full_metrics=True, cut_len=SR * 2,
    )

    print(f"\n  NISQA MOS: {results['NISQA MOS']:.4f}")
    print(f"  PESQ:      {results['PESQ']:.4f}")
    print(f"  SSNR:      {results['SSNR']:.2f} dB")
    print(f"  SI-SDR:    {results['SI-SDR']:.2f} dB")


if __name__ == "__main__":
    main()
