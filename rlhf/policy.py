import math

import torch

from loss import gaussian_log_prob, gaussian_kl

from models.cmgan.src.utils import power_compress, power_uncompress
from models.metricgan_plus.signal_processing import get_spec_and_phase, transform_spec_to_wav


def sample_noise(mean, sigma=0.01):
    return mean + torch.randn_like(mean) * sigma


# cmgan short time fourier transform
def cmgan_stft(noisy_wav):
    device = noisy_wav.device
    window = torch.hamming_window(400).to(device)

    c = torch.sqrt(noisy_wav.size(-1) / torch.sum(noisy_wav**2, dim=-1))
    noisy_norm = (noisy_wav.T*c).T

    noisy_spec = torch.stft(
        noisy_norm, 400, 100, window=window, onesided=True, return_complex=False
    )
    noisy_spec = power_compress(noisy_spec)(noisy_spec).permute(0, 1, 3, 2)

    return noisy_spec, c, window


# cmgan forward pass
def cmgan_forward(generator, noisy_wav, sigma=0.01):
    # short time fourier tansform
    noisy_spec, c, window  = cmgan_stft(noisy_wav)

    # forward pass
    est_real, est_imag = generator(noisy_spec)
    est_real = est_real.permute(0, 1, 3, 2)
    est_imag = est_imag.permute(0, 1, 3, 2)

    action_mean = (est_real, est_imag)

    # add gaussian noise
    noised_real = sample_noise(est_real, sigma)
    noised_imag = sample_noise(est_imag, sigma)
    action = (noised_real, noised_imag)

    # compute log probability
    lp = gaussian_log_prob(
        action = torch.cat([noised_real, noised_imag], diim=1),
        mean = torch.cat([est_real, est_imag], dim=1),
        sigma = sigma,
    )

    # uncompress
    est_spec = power_uncompress(noised_real, noised_imag).squeeze(1)

    # inverse short time fourier transform
    enhanced_wav = torch.istft(
        est_spec, 400, 100, window=window, onesided=True
    )

    return enhanced_wav, action_mean, action, lp


