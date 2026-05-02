# RLHF-SE

**Reinforcement Learning from Human Feedback for Speech Enhancement.**
A PyTorch implementation of [Using RLHF to align speech enhancement approaches to mean-opinion quality scores](https://arxiv.org/pdf/2410.13182), fine-tuning pre-trained speech enhancement models (CMGAN, MetricGAN+) with PPO to maximize predicted human-perceived quality (NISQA MOS).

## Why

Standard speech enhancement models optimize reference-based metrics like PESQ or STOI, which do not always reflect human perception. We follow the paper's RLHF approach: take a frozen pre-trained "SFT" generator, fine-tune a copy of it with PPO, and use NISQA — a neural MOS predictor — as the reward signal.

```
reward  =  NISQA(ŷ_rl) − NISQA(ŷ_sft)
loss    =  L_PPO-clip  +  λ · L_MSE
```

The KL penalty keeps the RL policy close to the SFT reference; the MSE term keeps it close to the clean target.

## Results

After 2000 steps on VoiceBank+DEMAND:

| Model      | Baseline MOS | RLHF (λ=1) |   Δ   | PPO-only ablation (λ=0) |
| ---------- | :----------: | :--------: | :---: | :---------------------: |
| CMGAN      |     3.28     |  **3.38**  | +0.10 |      3.29 (+0.01)       |
| MetricGAN+ |     4.11     |  **4.14**  | +0.03 |      4.14 (+0.03)       |

The ablation confirms the paper's claim: removing the MSE term (λ=0) collapses the gain on CMGAN, validating that MSE is necessary for stable fine-tuning.

**Training curves** (reward, PPO loss, MSE loss, total loss, test MOS):

|                         CMGAN                          |                           MetricGAN+                            |
| :----------------------------------------------------: | :-------------------------------------------------------------: |
| ![CMGAN curves](logs/rlhf/lambda%3D1/cmgan_curves.png) | ![MetricGAN+ curves](logs/rlhf/lambda%3D1/metricgan_curves.png) |

Per-step metrics CSVs are in `logs/rlhf/lambda=1/` (proposed approach) and `logs/rlhf/lambda=0/` (ablation).

## Setup

```bash
uv sync                                 # creates .venv and installs deps
uv run python train_rlhf.py             # runs RLHF fine-tuning
```

Or, with pip:
```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows; use source .venv/bin/activate on Mac/Linux
pip install -e .
python train_rlhf.py
```

Switch between models by editing the `MODEL` constant at the top of `train_rlhf.py` (`"cmgan"` or `"metricgan"`).

## Datasets

Place the dataset under `data/voicebank_demand/{train,test}/{clean,noisy}/`. Files are gitignored due to size.

- **VoiceBank+DEMAND (16 kHz):** [Kaggle](https://www.kaggle.com/datasets/jweiqi/voicebank-demand-16k)
- **LibriMix:** [GitHub](https://github.com/JorisCos/LibriMix)

## Project Structure

> `models/cmgan/`, `models/metricgan_plus/`, and `nisqa/` are cloned third-party
> repositories included unchanged. Original code lives in `train_rlhf.py` and `rlhf/`.

```
speech-enhancement-RLHF/
├── train_rlhf.py                # Training entry point (CMGAN + MetricGAN+ PPO loops)
├── third_party_path_setup.py    # sys.path setup for CMGAN's internal imports
├── rlhf/
│   ├── loss.py                  # PPO clip loss, KL divergence, MSE, combined loss
│   ├── policy.py                # Forward + log-prob recomputation (CMGAN, MetricGAN+)
│   ├── buffer.py                # Experience replay buffer
│   └── nisqa.py                 # NISQA reward model (load + batched MOS inference)
├── models/
│   ├── cmgan/                   # CMGAN: TSCNet generator (Conformer)
│   └── metricgan_plus/          # MetricGAN+: BiLSTM generator
├── nisqa/                       # NISQA neural MOS predictor (frozen reward model)
├── logs/rlhf/                   # Training metrics (CSV) and curves (PNG)
├── docs/
└── pyproject.toml               # Project dependencies
```

## Hyperparameters

Following [the paper](https://arxiv.org/pdf/2410.13182):

| Parameter        | Value  | Meaning                              |
| ---------------- | ------ | ------------------------------------ |
| ε (PPO clip)     | 0.01   | PPO step size limit                  |
| β (KL weight)    | 0.0001 | KL divergence penalty                |
| λ (MSE weight)   | 1.0    | Weight of MSE loss                   |
| α (MSE channels) | 0.7    | Magnitude vs. complex weight (CMGAN) |
| σ (Gaussian)     | 0.01   | Exploration noise on masks           |
| Learning rate    | 1e-6   |                                      |
| Effective batch  | 64     | 4×16 (CMGAN) or 8×8 (MetricGAN+)     |
| PPO epochs       | 5      | Per buffer fill                      |

## Citation

```bibtex
@misc{kumar2024usingrlhfalignspeech,
      title={Using RLHF to align speech enhancement approaches to mean-opinion quality scores},
      author={Anurag Kumar and Andrew Perrault and Donald S. Williamson},
      year={2024},
      eprint={2410.13182},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.13182},
}
```
