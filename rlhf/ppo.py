import torch


def compute_j_theta(reward, kl_div, beta=0.0001):
    return reward - beta * kl_div


def ppo_clip_loss(log_prob_new, log_prob_old, j_theta, eps=0.01):
    ratio = torch.exp(log_prob_new - log_prob_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    surrogate = torch.min(ratio * j_theta, clipped_ratio * j_theta)
    return -surrogate.mean()
