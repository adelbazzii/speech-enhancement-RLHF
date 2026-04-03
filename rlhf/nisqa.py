import numpy as np
import torch
import torchaudio

from nisqa.nisqa.NISQA_lib import NISQA

# NISQA was trained on 48kHz audio
NISQA_SR = 48000


def load_nisqa(ckpt_path, device):
    """
    Loads the NISQA MOS prediction model from a .tar checkpoint.
    The checkpoint contains both the model weights and its configuration.
    NISQA is frozen — it is only used as a reward signal, never trained.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["args"]

    nisqa = NISQA(
        ms_seg_length=model_args["ms_seg_length"],
        ms_n_mels=model_args["ms_n_mels"],
        cnn_model=model_args["cnn_model"],
        cnn_c_out_1=model_args["cnn_c_out_1"],
        cnn_c_out_2=model_args["cnn_c_out_2"],
        cnn_c_out_3=model_args["cnn_c_out_3"],
        cnn_kernel_size=model_args["cnn_kernel_size"],
        cnn_dropout=model_args["cnn_dropout"],
        cnn_pool_1=model_args["cnn_pool_1"],
        cnn_pool_2=model_args["cnn_pool_2"],
        cnn_pool_3=model_args["cnn_pool_3"],
        cnn_fc_out_h=model_args["cnn_fc_out_h"],
        td=model_args["td"],
        td_sa_d_model=model_args["td_sa_d_model"],
        td_sa_nhead=model_args["td_sa_nhead"],
        td_sa_pos_enc=model_args["td_sa_pos_enc"],
        td_sa_num_layers=model_args["td_sa_num_layers"],
        td_sa_h=model_args["td_sa_h"],
        td_sa_dropout=model_args["td_sa_dropout"],
        td_lstm_h=model_args["td_lstm_h"],
        td_lstm_num_layers=model_args["td_lstm_num_layers"],
        td_lstm_dropout=model_args["td_lstm_dropout"],
        td_lstm_bidirectional=model_args["td_lstm_bidirectional"],
        td_2=model_args["td_2"],
        td_2_sa_d_model=model_args["td_2_sa_d_model"],
        td_2_sa_nhead=model_args["td_2_sa_nhead"],
        td_2_sa_pos_enc=model_args["td_2_sa_pos_enc"],
        td_2_sa_num_layers=model_args["td_2_sa_num_layers"],
        td_2_sa_h=model_args["td_2_sa_h"],
        td_2_sa_dropout=model_args["td_2_sa_dropout"],
        td_2_lstm_h=model_args["td_2_lstm_h"],
        td_2_lstm_num_layers=model_args["td_2_lstm_num_layers"],
        td_2_lstm_dropout=model_args["td_2_lstm_dropout"],
        td_2_lstm_bidirectional=model_args["td_2_lstm_bidirectional"],
        pool=model_args["pool"],
        pool_att_h=model_args["pool_att_h"],
        pool_att_dropout=model_args["pool_att_dropout"],
    )

    nisqa.load_state_dict(checkpoint["model_state_dict"])
    nisqa.to(device)
    nisqa.eval()
    for p in nisqa.parameters():
        p.requires_grad = False

    return nisqa, model_args


def _build_mel_transform(model_args, device):
    """Build a torchaudio MelSpectrogram."""
    hop_length = int(NISQA_SR * model_args.get("ms_hop_length", 0.01))
    win_length = int(NISQA_SR * model_args.get("ms_win_length", 0.02))

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=NISQA_SR,
        n_fft=model_args.get("ms_n_fft", 4096),
        hop_length=hop_length,
        win_length=win_length,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=model_args.get("ms_n_mels", 48),
        f_min=0.0,
        f_max=model_args.get("ms_fmax", 20000),
        norm="slaney",
        mel_scale="slaney",
    ).to(device)


def _amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0):
    """Convert amplitude spectrogram to dB."""
    magnitude = torch.clamp(S, min=amin)
    db = 20.0 * torch.log10(magnitude / ref)
    if top_db is not None:
        db = torch.clamp(db, min=db.max() - top_db)
    return db


def _segment_spec(spec, seg_length, seg_hop, max_segments):
    """Segment a mel-spectrogram into overlapping windows.

    Args:
        spec: [n_mels, T] mel-spectrogram tensor.
        seg_length: segment width in time bins.
        seg_hop: hop between segments in time bins.
        max_segments: pad output to this length.

    Returns:
        (x_seg, n_wins) where x_seg is [max_segments, 1, n_mels, seg_length].
    """
    _, T = spec.shape

    # pad if too short
    if T < seg_length:
        spec = torch.nn.functional.pad(spec, (0, seg_length - T))
        T = spec.shape[1]

    n_wins = T - (seg_length - 1)

    # sliding window: [n_wins, seg_length]
    idx1 = torch.arange(seg_length, device=spec.device)
    idx2 = torch.arange(n_wins, device=spec.device)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)

    # spec.T is [T, n_mels] -> index -> [n_wins, seg_length, n_mels]
    # then -> [n_wins, 1, n_mels, seg_length]
    x = spec.T[idx3, :]
    x = x.unsqueeze(1).permute(0, 1, 3, 2)

    # apply segment hop
    if seg_hop > 1:
        x = x[::seg_hop]
        n_wins = int(np.ceil(n_wins / seg_hop))

    # pad to max_segments
    actual = x.shape[0]
    if actual < max_segments:
        pad = torch.zeros(
            max_segments - actual,
            x.shape[1],
            x.shape[2],
            x.shape[3],
            device=spec.device,
        )
        x = torch.cat([x, pad], dim=0)
    else:
        x = x[:max_segments]
        n_wins = min(n_wins, max_segments)

    return x, n_wins


@torch.no_grad()
def get_nisqa_score(nisqa_model, model_args, audio_tensor, device, fs=16000):
    """
    Computes the NISQA MOS score for a batch of audio tensors.

    Args:
        nisqa_model: loaded NISQA model.
        model_args: checkpoint args dict.
        audio_tensor: [B, L] waveform tensor at sample rate fs.
        device: torch device.
        fs: input sample rate (default 16000).

    Returns:
        Tensor of shape [B] with one MOS score per sample.
    """
    seg_length = model_args.get("ms_seg_length", 15)
    seg_hop = model_args.get("ms_seg_hop_length", 4)
    max_segments = model_args.get("ms_max_segments", 1300)

    mel_transform = _build_mel_transform(model_args, device)
    resampler = torchaudio.transforms.Resample(fs, NISQA_SR).to(device)

    audio_tensor = audio_tensor.to(device)

    # resample to 48kHz
    if fs != NISQA_SR:
        audio_tensor = resampler(audio_tensor)

    # preprocess all samples
    all_specs = []
    all_n_wins = []
    for i in range(audio_tensor.shape[0]):
        mel = mel_transform(audio_tensor[i].unsqueeze(0)).squeeze(0)
        mel_db = _amplitude_to_db(mel)
        x_seg, n_wins = _segment_spec(mel_db, seg_length, seg_hop, max_segments)
        all_specs.append(x_seg)
        all_n_wins.append(n_wins)

    # batch and run NISQA in one forward pass
    x_batch = torch.stack(all_specs)  # [B, max_segments, 1, n_mels, seg_length]
    n_wins_batch = torch.tensor(all_n_wins, dtype=torch.long, device=device)
    out = nisqa_model(x_batch, n_wins_batch)  # [B, 1]
    return out[:, 0]  # [B]
