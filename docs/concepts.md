# Project Concepts & Design Notes

This document covers the three repositories used in this project, key ML concepts, and the RLHF design decisions derived from the paper.

**Paper:** [Using RLHF to align speech enhancement approaches to mean-opinion quality scores](https://arxiv.org/pdf/2410.13182)

---

## The Three Cloned Repositories

### 1. CMGAN (models/cmgan/)

CMGAN (Conformer-based Metric GAN) is a speech enhancement model based on a GAN architecture. It takes a noisy audio input and produces a clean enhanced audio output.

**Architecture:**
- **Generator (TSCNet):** Takes the noisy spectrogram and predicts magnitude and complex (real + imaginary) masks. Uses 4 Two-Stage Conformer blocks to process the spectrogram along both time and frequency axes.
- **Discriminator:** A CNN with spectral normalization that scores the quality of the enhanced audio by comparing it to the clean reference. Trained to predict real PESQ scores.

**Training:** Distributed multi-GPU training (DDP). The generator loss is a weighted combination of:
- Real/imaginary spectrogram error
- Magnitude error
- Time-domain waveform error
- Discriminator score

**Key files:**
| File | Purpose |
|------|---------|
| `src/models/generator.py` | TSCNet generator |
| `src/models/discriminator.py` | GAN discriminator |
| `src/models/conformer.py` | Conformer blocks |
| `src/data/dataloader.py` | VoiceBank+DEMAND data loader |
| `src/train.py` | Supervised training loop |
| `src/evaluation.py` | Inference and metric evaluation |
| `src/tools/compute_metrics.py` | PESQ, STOI, CSIG, CBAK, COVL, SSNR |

---

### 2. MetricGAN+ (`models/metricgan_plus/`)

MetricGAN+ is an LSTM-based speech enhancement model. Unlike CMGAN, it only predicts a magnitude mask (no complex components).

**Architecture:**
- **Generator:** 2-layer Bidirectional LSTM followed by fully connected layers. Outputs a magnitude mask applied to the noisy spectrogram.
- **Discriminator:** Same LSTM architecture, trained to predict quality metric scores (PESQ, CSIG, CBAK, or COVL - configurable).

**Training:** Alternates each epoch between two phases:
1. **Generator phase:** Enhance audio → ask discriminator to score it → train generator to maximize that score
2. **Discriminator phase:** Compute real metric scores on enhanced/noisy/clean audio → train discriminator to predict them accurately

Also uses a **replay buffer** (`historical_set`) — mixes past enhanced samples into discriminator training to prevent catastrophic forgetting (a common GAN issue).

**Key files:**
| File | Purpose |
|------|---------|
| `model.py` | Generator + Discriminator (BiLSTM) |
| `train.py` | Training loop with replay buffer |
| `inference.py` | Inference pipeline |
| `dataloader.py` | Data loading |
| `signal_processing.py` | STFT, magnitude extraction, iSTFT |
| `metric_functions/` | PESQ, CSIG, CBAK, COVL scoring |

---

### 3. NISQA (`nisqa/`)

NISQA (Non-Intrusive Speech Quality Assessment) is a neural network that predicts the MOS (Mean Opinion Score) of a speech signal **without needing a clean reference**. It is used as the reward model in RLHF fine-tuning.

**Architecture:** CNN for frame-level feature extraction, followed by Self-Attention or LSTM for temporal modeling, and attention-based pooling.

**Outputs:**
- `nisqa_mos_only.tar` — predicts overall MOS score only (used as reward)
- `nisqa.tar` — predicts MOS + 4 dimensions: noisiness, coloration, discontinuity, loudness
- `nisqa_tts.tar` — fine-tuned for TTS audio (not used here)

**Key files:**
| File | Purpose |
|------|---------|
| `nisqa/NISQA_lib.py` | Core model implementation |
| `nisqa/NISQA_model.py` | High-level training/inference wrapper |
| `run_predict.py` | Predict MOS score for audio files |
| `weights/` | Pre-trained weight files (.tar) |

---

## Key ML Concepts

### Generator
The model that does the actual work: takes noisy audio and outputs enhanced (denoised) audio. In CMGAN it is TSCNet, in MetricGAN+ it is a BiLSTM. This is the model being fine-tuned with RLHF.

### Discriminator
A second model that acts as a judge — it scores the quality of enhanced audio. During supervised pre-training of CMGAN and MetricGAN+, it is trained to predict real metric scores (e.g. PESQ), and the generator is trained to fool it into giving high scores.

> The discriminator is only used during supervised pre-training. It plays no role during RLHF fine-tuning — NISQA replaces it as the reward signal.

### Inference
Running a trained model on new audio to produce enhanced output — no training, no gradient updates, just a forward pass. The inference code does not change; only the model weights do.

### Data Loader
Code that reads audio files from disk, slices them into fixed-length segments, batches them, and feeds them to the model during training.

### Conformer
A neural network block combining:
- **Transformer** (self-attention) — captures global patterns across the full audio sequence
- **Convolution** — captures local patterns in nearby time/frequency frames

The name comes from **Con**volution + Trans**former**. CMGAN uses 4 Conformer blocks along both time and frequency axes — hence Two-Stage Conformer Network (TSCNet).

### MOS (Mean Opinion Score)
A standard human-rated speech quality metric. Listeners rate audio from 1 (bad) to 5 (excellent). NISQA predicts what that human score would be without requiring real listeners.

### Other Objective Metrics
| Metric | What it measures |
|--------|-----------------|
| PESQ | Perceptual Evaluation of Speech Quality |
| CSIG | Signal distortion perceived by the listener |
| CBAK | Background noise intrusiveness |
| COVL | Overall quality (combines CSIG and CBAK) |
| STOI | Short-Time Objective Intelligibility |
| SSNR | Segmental Signal-to-Noise Ratio |
| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio |

---

## RLHF Training Framework

### Why RLHF?
CMGAN and MetricGAN+ use objective metrics (PESQ, etc.) as training targets. These do not always align with how humans perceive quality. RLHF addresses this by using a MOS-based reward (NISQA) to fine-tune the already pre-trained generator toward better human-perceived quality.

### Supervised (GAN) Training vs RLHF Fine-tuning

| x | GAN supervised training | RLHF fine-tuning |
|---|---|---|
| Training signal | Gradient via backprop through discriminator | Scalar reward via PPO |
| Differentiable? | Yes | No — NISQA is a black box to the generator |
| Paradigm | Supervised / adversarial | Reinforcement learning |
| Loss | MSE + discriminator loss | PPO clip loss + MSE loss |

### Reward Signal
The reward is the **relative improvement** in NISQA MOS between the RL policy and the frozen SFT (supervised fine-tuning) policy:

```
r_mos = NISQA(ŷ_rl) − NISQA(ŷ_sft)
```

The generator is only rewarded when it does *better than* the frozen reference — not in absolute terms. The paper also tried `r_comb = r_mos + r_pesq` but it performed worse than either reward individually.

### Loss Function (Equation 5 in the paper)
```
L_theta = L_ppo-clip + λ × L_MSE
```

- **PPO clip loss** — policy gradient loss that maximizes expected reward while preventing large policy updates
- **MSE loss** — reconstruction loss that keeps the model close to the original supervised solution and helps generalization
- **λ = 1.0**

The MSE loss is necessary — the paper's ablation study shows that using PPO alone leads to poor generalization.

### KL Divergence
Inside the PPO objective, a KL divergence penalty keeps the RL policy from drifting too far from the SFT policy:

```
J_theta = reward − β × KL(π_rl, π_sft)
```

**β = 0.0001**

### Key Hyperparameters (from the paper)
| Parameter | Value | Meaning |
|-----------|-------|---------|
| ε (PPO clip) | 0.01 | Step size limit for PPO update |
| β (KL weight) | 0.0001 | KL divergence penalty weight |
| λ (MSE weight) | 1.0 | Weight of MSE loss |
| α (MSE channels) | 0.7 | Weight of magnitude vs complex components in MSE |
| σ (Gaussian noise) | 0.01 | Noise added to masks for RL exploration |
| Learning rate | 1e-6 | Very small — fine-tuning only |
| Batch size | 4 (CMGAN), 8 (MetricGAN+) | Gradient accumulation → effective batch = 64 |

### Results (from the paper, VoiceBank+DEMAND)
| Model | PESQ | SSNR | SI-SDR | NISQA MOS |
|-------|------|------|--------|-----------|
| CMGAN | 3.39 | 10.35 | 20.02 | 4.65 |
| CMGAN_PPO | 3.39 | 10.93 | 20.28 | 4.73 |
| MetricGAN+ | 3.15 | 3.11 | 7.98 | 3.99 |
| MetricGAN_PPO | 3.12 | 3.51 | 9.57 | 4.08 |

RLHF fine-tuning improves SSNR, SI-SDR, and NISQA MOS without degrading PESQ or STOI.

---

*Last edited: 2026-04-03*
