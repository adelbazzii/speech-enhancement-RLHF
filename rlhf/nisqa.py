import os
import tempfile

import numpy as np
import soundfile as sf
import torch

from nisqa.nisqa.NISQA_lib import NISQA, get_librosa_melspec


def load_nisqa(ckpt_path, device):
    """
    Loads the NISQA MOS prediction model from a .tar checkpoint.
    The checkpoint contains both the model weights and its configuration.
    NISQA is frozen — it is only used as a reward signal, never trained.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
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


def get_nisqa_score(nisqa_model, model_args, audio_tensor, device, fs=16000):
    """
    Computes the NISQA MOS score for a batch of audio tensors.
    NISQA expects mel-spectrograms computed from audio files, so we save
    each waveform to a temporary file, compute the mel-spectrogram, then
    run the model.

    Returns a tensor of shape [B] with one MOS score per sample.
    """
    seg_length = model_args["ms_seg_length"]
    scores = []

    for i in range(audio_tensor.shape[0]):
        wav = audio_tensor[i].detach().cpu().numpy()

        # Save to a temp file so NISQA's melspec function can load it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, wav, fs)

        # Compute mel-spectrogram using NISQA's own preprocessing
        spec = get_librosa_melspec(
            tmp_path,
            sr=model_args["ms_sr"],
            n_fft=model_args["ms_n_fft"],
            hop_length=model_args["ms_hop_length"],
            win_length=model_args["ms_win_length"],
            n_mels=model_args["ms_n_mels"],
            fmax=model_args["ms_fmax"],
        )
        os.remove(tmp_path)

        # Segment the spectrogram into fixed-length windows (shape: [n_wins, seg_length, n_mels])
        n_frames = spec.shape[1]
        if n_frames < seg_length:
            pad = seg_length - n_frames
            spec = np.pad(spec, ((0, 0), (0, pad)), mode="constant")
            n_frames = seg_length

        n_wins = n_frames - seg_length + 1
        segments = np.stack([spec[:, j:j + seg_length].T for j in range(n_wins)])  # [n_wins, seg_length, n_mels]
        x = torch.tensor(segments, dtype=torch.float32).unsqueeze(0).to(device)    # [1, n_wins, seg_length, n_mels]
        n_wins_tensor = torch.tensor([n_wins], dtype=torch.long).to(device)

        with torch.no_grad():
            mos = nisqa_model(x, n_wins_tensor)  # [1, 1]
        scores.append(mos.squeeze())

    return torch.stack(scores)  # [B]
