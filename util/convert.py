import librosa
import numpy as np
import soundfile as sf
import torch.nn as nn
from melgan.interface import *

vocoder = MelVocoder(model_name="model/static/multi_speaker")


def get_mel(mels, seq_len, device):
    # input -> (len,80)
    mel, tmp_m = [], []
    pad_size, last_len = 0, 0
    for i in range(0, mels.shape[0], seq_len):
        if i + seq_len <= mels.shape[0]:
            mel.append(
                torch.from_numpy(mels[i : i + seq_len, :]).unsqueeze(0).to(device)
            )
            last_len = i + seq_len

    if (mels.shape[0] - last_len) > 30:
        pad_size = int((last_len + seq_len) - mels.shape[0])
        npad = [(0, 0)] * 2
        npad[0] = (0, pad_size)
        tmp_m = np.pad(
            mels[last_len:, :], pad_width=npad, mode="constant", constant_values=0
        )
        mel.append(torch.from_numpy(tmp_m).unsqueeze(0).to(device))

    return mel, pad_size


def get_trans_mel(
    model: nn.Module,
    mel_source,
    mel_target,
    emb_org,
    emb_trg,
    isAdain: bool,
    isVQ: bool,
    isAgainVC: bool,
    seq_len=128,
    device="cuda:0",
):
    waves = []

    source_mels, pad_size_source = get_mel(mel_source, seq_len, device)
    if isVQ or isAgainVC:
        target_mels, _ = get_mel(mel_target, seq_len, device)

    else:
        target_mels = [0] * len(source_mels)

    last_ = len(source_mels)
    for j, mels in enumerate(zip(source_mels, target_mels)):
        mel_source, mel_target = mels
        if isAdain:
            _, feature = model(mel_source, emb_org, None, None)
            _, mel_trans, _ = model(mel_source, emb_org, emb_trg, feature)
        if isVQ:
            q_after_block, sp_embedding_block, std_block, _ = model.encode(
                mel_source.transpose(1, 2)
            )
            q_after_block_tg, sp_embedding_block_tg, std_block_tg, _ = model.encode(
                mel_target.transpose(1, 2)
            )
            mel_trans = model.decode(
                q_after_block, sp_embedding_block_tg, std_block_tg
            ).transpose(1, 2)
        elif isAgainVC:
            mel_source = mel_source.transpose(1, 2)
            mel_target = mel_target.transpose(1, 2)
            mel_trans = model.inference(mel_source, mel_target).transpose(1, 2)
        else:
            _, mel_trans, _ = model(mel_source, emb_org, emb_trg)
        mel_trans = mel_trans.squeeze(1)

        if pad_size_source > 0 and (j + 1) == last_:
            mel_source = mel_source[:, : (seq_len - pad_size_source), :]
            mel_trans = mel_trans[:, : (seq_len - pad_size_source), :]
        waves.append(vocoder.inverse(mel_trans.transpose(2, 1)).squeeze())

    return librosa.effects.trim(np.hstack(waves), top_db=20)[0]


def convert(mel):
    wave = vocoder.inverse(mel.transpose(2, 1)).squeeze()
    return librosa.effects.trim(np.hstack(wave), top_db=20)[0]


def save_wave(name, wave, sr):
    sf.write(name, wave, sr)
