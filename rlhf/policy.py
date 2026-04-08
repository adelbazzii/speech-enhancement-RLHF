import torch
from rlhf.loss import gaussian_log_prob

from models.cmgan.src.utils import power_uncompress
from models.metricgan_plus.signal_processing import transform_spec_to_wav


def sample_noise(mean, sigma=0.01):
    return mean + torch.randn_like(mean) * sigma


# cmgan forward pass
def cmgan_forward(generator, noisy_spec, window, sigma=0.01, add_noise=True):
    device = noisy_spec.device

    # forward pass
    est_real, est_imag = generator(noisy_spec)
    est_real = est_real.permute(0, 1, 3, 2)
    est_imag = est_imag.permute(0, 1, 3, 2)

    action_mean = (est_real, est_imag)

    if add_noise:
        # add gaussian noise
        noised_real = sample_noise(est_real, sigma)
        noised_imag = sample_noise(est_imag, sigma)
        action = (noised_real, noised_imag)

        # compute log probability
        lp = gaussian_log_prob(
            action=torch.cat([noised_real, noised_imag], dim=1),
            mean=torch.cat([est_real, est_imag], dim=1),
            sigma=sigma,
        )
    else:
        action = action_mean
        noised_real, noised_imag = est_real, est_imag
        lp = torch.zeros(noisy_spec.shape[0], device=device)

    # uncompress
    est_spec = power_uncompress(noised_real, noised_imag).squeeze(1)

    # inverse short time fourier transform
    enhanced_wav = torch.istft(est_spec, 400, 100, window=window, onesided=True)

    return enhanced_wav, action_mean, action, lp


def metricgan_forward(generator, noise_mag, noise_phase, sigma=0.01, add_noise=True, signal_length=None):
    device = noise_mag.device

    mask_mean = generator(noise_mag).clamp(min=0.05)

    if add_noise:
        noised_mask = sample_noise(mask_mean, sigma).clamp(min=0.05)
        action = noised_mask
        lp = gaussian_log_prob(action=noised_mask, mean=mask_mean, sigma=sigma)
    else:
        action = mask_mean
        noised_mask = mask_mean
        lp = torch.zeros(noise_mag.shape[0], device=device)

    enh_mag = noised_mask * noise_mag
    enh_audio = transform_spec_to_wav(torch.expm1(enh_mag), noise_phase, signal_length)

    return enh_audio, mask_mean, action, lp


# recompute log probabilities for current policy
def cmgan_recompute_log_prob(generator, noisy_spec, stored_action, sigma=0.01):
    est_real, est_imag = generator(noisy_spec)
    est_real = est_real.permute(0, 1, 3, 2)
    est_imag = est_imag.permute(0, 1, 3, 2)

    stored_real, stored_imag = stored_action
    lp = gaussian_log_prob(
        action=torch.cat([stored_real, stored_imag], dim=1),
        mean=torch.cat([est_real, est_imag], dim=1),
        sigma=sigma,
    )

    return lp, (est_real, est_imag)


def metricgan_recompute_log_prob(generator, noise_mag, stored_action, sigma=0.01):
    mask_mean = generator(noise_mag).clamp(min=0.05)
    lp = gaussian_log_prob(action=stored_action, mean=mask_mean, sigma=sigma)
    enh_mag = stored_action * noise_mag
    return lp, enh_mag
