# RLHF-SE

### Paper

Using RLHF to align speech enhancement approaches to mean-opinion quality scores

[https://arxiv.org/pdf/2410.13182](https://arxiv.org/pdf/2410.13182)

---

### Datasets

Dataset placeholder folders are added. Data are ignored by git due to their large size.

Datasets used are referenced below.

**a) VoiceBank_DEMAND_16k**

Source: [https://www.kaggle.com/datasets/jweiqi/voicebank-demand-16k](https://www.kaggle.com/datasets/jweiqi/voicebank-demand-16k)

**b) LibriMix**

Source: [https://github.com/JorisCos/LibriMix](https://github.com/JorisCos/LibriMix)

---

### Setup

**Using uv (recommended):**

```bash
uv sync
```

**Using pip:**

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -e .
```

---

### Project Structure

> **Note:** `models/cmgan/`, `models/metricgan_plus/`, and `nisqa/` are cloned
> third-party repositories included unchanged. Original code lives in `train_rlhf.py`
> and `rlhf/`.

```
speech-enhancement-RLHF/
│
├── train_rlhf.py                # RLHF training entry point (in progress)
├── rlhf/
│   └── loss.py                  # PPO, KL divergence, and MSE loss functions
│
├── models/
│   ├── cmgan/                   # cloned: Conformer-based Metric GAN (CMGAN)
│   │   └── src/
│   │       ├── models/          #   Generator (TSCNet), Discriminator, Conformer blocks
│   │       ├── data/            #   VoiceBank+DEMAND data loader
│   │       ├── train.py         #   Supervised training (distributed)
│   │       ├── evaluation.py    #   Inference + metric evaluation
│   │       └── tools/           #   PESQ, STOI, CSIG, CBAK, COVL metrics
│   │
│   └── metricgan_plus/          # cloned: MetricGAN+ (LSTM-based enhancement)
│       ├── model.py             #   Generator + Discriminator (BiLSTM)
│       ├── train.py             #   Training loop with replay buffer
│       ├── inference.py         #   Inference pipeline
│       ├── signal_processing.py #   STFT and phase handling
│       └── metric_functions/    #   PESQ, CSIG, CBAK, COVL scoring
│
├── nisqa/                       # cloned: NISQA neural speech quality predictor
│   ├── nisqa/NISQA_lib.py       #   Core model (CNN + Self-Attention/LSTM)
│   ├── nisqa/NISQA_model.py     #   High-level training/inference wrapper
│   ├── run_predict.py           #   Predict MOS score for audio files
│   ├── config/                  #   YAML training configurations
│   └── weights/                 #   Pre-trained model weights (.tar files)
│
├── data/
│   ├── voicebank_demand/        # VoiceBank+DEMAND dataset (not tracked by git)
│   └── libri_mix/               # LibriMix dataset placeholder (not tracked by git)
│
├── pyproject.toml               # Project dependencies
└── uv.lock                      # Pinned dependency versions (uv lockfile)
```

---

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| CMGAN model | Cloned | TSCNet generator, discriminator, supervised training |
| MetricGAN+ model | Cloned | BiLSTM generator, discriminator, training |
| NISQA reward model | Cloned | CNN+SA quality predictor, predict/train scripts |
| RLHF loss functions (`rlhf/loss.py`) | Complete | PPO clip loss, KL divergence, combined loss |
| RLHF training loop (`train_rlhf.py`) | In progress | CMGAN loop complete; MetricGAN+ stub remaining |

---

### What Remains To Be Implemented

**train_rlhf.py - CMGAN**

- [x] SFT policy forward pass: STFT → normalize → run frozen generator → get masks
- [x] RL policy forward pass: same pipeline + add Gaussian noise to masks
- [x] Convert both outputs back to audio (iSTFT) for reward computation
- [x] Load NISQA from `nisqa/weights/nisqa_mos_only.tar` and run inference on both outputs
- [x] Compute reward: `r_mos = NISQA(ŷ_rl) − NISQA(ŷ_sft)`
- [x] Compute KL divergence between RL and SFT policies using `gaussian_kl()`
- [x] Compute `J_theta = reward − β × KL` using `compute_j_theta()`
- [x] Compute log probabilities for old and new RL policy using `gaussian_log_prob()`
- [x] Compute PPO clip loss using `ppo_clip_loss()`
- [x] Compute MSE loss using `cmgan_mse_loss()`
- [x] Combine: `L_theta = ppo_loss + λ × mse_loss` using `combined_loss()`
- [x] Gradient accumulation and optimizer step

**train_rlhf.py - MetricGAN+**

- [ ] Implement `train_metricgan_rlhf()` (currently empty stub) following same RLHF loop
- [ ] Use `metricgan_mse_loss()` instead of `cmgan_mse_loss()`

**train_rlhf.py - General**

- [x] Implement `main()` with hardcoded configuration
- [ ] Checkpoint saving for RLHF-trained models
- [ ] Logging (loss, reward, KL divergence per step)
- [ ] Evaluation against baseline after training

---

*Last edited: 2026-04-03*

