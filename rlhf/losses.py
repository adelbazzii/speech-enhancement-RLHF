import torch
import torch.nn.functional as F


def cmgan_mse_loss(est_real, est_imag, clean_real, clean_imag, alpha=0.7):
    est_mag = torch.sqrt(est_real**2 + est_imag**2)
    clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

    loss_mag = F.mse_loss(est_mag, clean_mag)
    loss_real = F.mse_loss(est_real, clean_real)
    loss_imag = F.mse_loss(est_imag, clean_imag)

    return alpha * loss_mag + (1 - alpha) * (loss_real + loss_imag)


def metricgan_mse_loss(enhanced_mag, clean_mag):
    return F.mse_loss(enhanced_mag, clean_mag)


def combined_loss(ppo_loss, mse_loss, lam=1.0):
    return ppo_loss + lam * mse_loss
