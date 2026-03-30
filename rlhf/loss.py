import math

import torch
import torch.nn.functional as F


def gaussian_log_prob(action, mean, sigma=0.01):
    B = action.shape[0]
    D = action.numel() / B
    diff_sq = ((action - mean) ** 2).view(B, -1).sum(dim=1)
    return -0.5 * diff_sq / (sigma**2) - 0.5 * D * math.log(2 * math.pi * sigma**2)


def gaussian_kl(mean_rl, mean_sft, sigma=0.01):
    B = mean_rl.shape[0]
    diff_sq = ((mean_rl - mean_sft) ** 2).view(B, -1).sum(dim=1)
    return diff_sq / (2 * sigma**2)


def compute_j_theta(reward, kl_div, beta=0.0001):
    return reward - beta * kl_div


def ppo_clip_loss(log_prob_new, log_prob_old, j_theta, eps=0.01):
    ratio = torch.exp(log_prob_new - log_prob_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    surrogate = torch.min(ratio * j_theta, clipped_ratio * j_theta)
    return -surrogate.mean()


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
