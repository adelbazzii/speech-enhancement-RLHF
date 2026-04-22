import os

import matplotlib.pyplot as plt
import pandas as pd

MODEL = "cmgan" # "cmgan" or "metricgan"
METRICS_PATH = f"logs/rlhf/lambda=0/{MODEL}_metrics.csv"
OUT_PATH = f"logs/rlhf/lambda=0/{MODEL}_curves.png"

METRICS = [
    ("reward", "Reward (rl_mos - sft_mos)"),
    ("ppo_loss", "PPO Loss"),
    ("mse_loss", "MSE Loss"),
    ("total_loss", "Total Loss"),
    ("test_mos", "Test NISQA MOS"),
]


def main():
    df = pd.read_csv(METRICS_PATH)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 2.4 * len(METRICS)), sharex=True)
    for ax, (col, label) in zip(axes, METRICS):
        series = df[["step", col]].dropna()
        ax.plot(series["step"], series[col], marker=".", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{MODEL.upper()} RLHF training curves")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
