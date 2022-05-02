import argparse
import csv
import importlib
import json
import pickle

import librosa
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import auc

from thirdparty.DeepSpeaker.util.audio import get_mfcc_from_wave
from thirdparty.DeepSpeaker.util.batcher import sample_from_mfcc
from thirdparty.DeepSpeaker.util.constants import NUM_FRAMES, SAMPLE_RATE
from thirdparty.DeepSpeaker.util.conv_models import DeepSpeakerModel
from thirdparty.DeepSpeaker.util.test import batch_cosine_similarity
from util.convert import convert, get_trans_mel


def generate_all_dv(embedder, metadata, erroment_num):
    all_dv = {}
    for data in metadata:
        print(f"Generate real embedding of {data[0]}")
        _dv = np.zeros((1, 512))
        for sound_id in range(erroment_num):
            source_path = data[sound_id + 2].replace("\\", "/")
            wave = convert(
                torch.from_numpy(np.load(f"{ROOT}/{source_path}")).unsqueeze(0)
            )
            # since melgan generate 22.05 khz voice
            wave = librosa.resample(wave, 22050, SAMPLE_RATE)
            mfcc = sample_from_mfcc(get_mfcc_from_wave(wave, SAMPLE_RATE), NUM_FRAMES)
            _dv += embedder.m.predict(np.expand_dims(mfcc, axis=0))
        all_dv[data[0]] = (_dv / erroment_num).reshape(-1)
    return all_dv


def generate_result(
    VC, embedder, metadata, sound_id, all_dv, is_adain, is_vq, is_again_vc
):
    cos_result = np.zeros((len(all_dv), len(all_dv), len(all_dv)))
    for i, datas in enumerate(metadata):
        print(f"Now Processing --- {datas[0]}")
        source_path = datas[sound_id + 2].replace("\\", "/")
        mel_source = np.load(f"{ROOT}/{source_path}")
        emb_org = torch.from_numpy(datas[1]).unsqueeze(0).to(device)

        for j, datat in enumerate(metadata):
            target_path = datat[sound_id + 2].replace("\\", "/")
            mel_target = np.load(f"{ROOT}/{target_path}")
            emb_trg = torch.from_numpy(datat[1]).unsqueeze(0).to(device)
            if i == j:
                print(f"Reconstruct ---- {datas[0]} to {datat[0]}")
            else:
                print(f"Trans --- {datas[0]} to {datat[0]}")

            wave = get_trans_mel(
                VC,
                mel_source,
                mel_target,
                emb_org,
                emb_trg,
                is_adain,
                is_vq,
                is_again_vc,
            )
            wave = librosa.resample(wave, 22050, SAMPLE_RATE)
            mfcc = sample_from_mfcc(get_mfcc_from_wave(wave, SAMPLE_RATE), NUM_FRAMES)
            _dv_result = embedder.m.predict(np.expand_dims(mfcc, axis=0))
            for k, dict_ in enumerate(all_dv.items()):
                _, emb_truth_dv = dict_
                cos = batch_cosine_similarity(emb_truth_dv, _dv_result)[0]
                cos_result[i][j][k] = cos
    return cos_result


def get_cos(cos_res):
    convert = []
    convert_ = []
    N = len(cos_res)
    for i in range(N):
        print(cos_res[i])
        convert_cos = np.sum(np.diagonal(cos_res[i]))
        convert.append(convert_cos / N)
        convert_.append(
            (np.sum(cos_res[i]) - np.sum(np.diagonal(cos_res[i]))) / ((N - 1) * N)
        )

    return sum(convert) / len(convert), sum(convert_) / len(convert_)


def get_ytrue_yscore(cos_res):
    N = len(cos_res)
    y_true, y_score = [], []
    for i in range(N):
        for ele in np.diagonal(cos_res[i]):
            y_score.append(ele)
        for _ in range(len(cos_res[i])):
            y_true.append(1)
        # remove diagonal element
        m = cos_res[i].shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = cos_res[i].strides
        outs = (
            strided(cos_res[i].ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1))
            .reshape(m, -1)
            .flatten()
        )
        for ele in outs:
            y_score.append(ele)
        for _ in range(len(outs)):
            y_true.append(0)
    return y_true, y_score


if __name__ == "__main__":
    json_config = json.load(open("config.json"))
    num_speaker = json_config["num_speaker"]
    erroment_num = json_config["erroment_num"]
    num_valid = json_config["num_valid"]
    dim_neck = json_config["dim_neck"]
    dim_emb = json_config["dim_emb"]
    dim_pre = json_config["dim_pre"]
    freq = json_config["freq"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="train_spmel_vctk80", required=False)
    parser.add_argument("--is_adain", default=False)
    parser.add_argument("--is_vq", default=False)
    parser.add_argument("--is_againvc", default=False)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pt_name", required=True)

    args = parser.parse_args()

    ROOT = "train_spmel_vctk80"
    embedder = DeepSpeakerModel()
    embedder.m.load_weights(
        f"thirdparty/DeepSpeaker/model/ResCNN_triplet_training_checkpoint_265.h5",
        by_name=True,
    )

    if bool(args.is_vq):
        VC = getattr(
            importlib.import_module(f"factory.{args.model_name}"), args.model_name
        )(80, 64, 64).to("cuda:0")
    elif bool(args.is_againvc):
        build_config = {
            "model_name": "again",
            "model": {
                "params": {
                    "encoder_params": {
                        "c_in": 80,
                        "c_h": 256,
                        "c_out": 4,
                        "n_conv_blocks": 6,
                        "subsample": [1, 1, 1, 1, 1, 1],
                    },
                    "decoder_params": {
                        "c_in": 4,
                        "c_h": 256,
                        "c_out": 80,
                        "n_conv_blocks": 6,
                        "upsample": [1, 1, 1, 1, 1, 1],
                    },
                    "activation_params": {"act": "sigmoid", "params": {"alpha": 0.1}},
                }
            },
            "optimizer": {
                "params": {
                    "lr": 0.0005,
                    "betas": [0.9, 0.999],
                    "amsgrad": True,
                    "weight_decay": 0.0001,
                },
                "grad_norm": 3,
            },
        }
        VC = getattr(
            importlib.import_module(f"factory.{args.model_name}"), args.model_name
        )(**build_config["model"]["params"]).to("cuda:0")
    else:
        VC = getattr(
            importlib.import_module(f"factory.{args.model_name}"), args.model_name
        )(dim_neck, dim_emb, dim_pre, freq).to("cuda:0")
    VC.load_state_dict(torch.load(f"model/{args.pt_name}.pt", map_location="cuda:0"))

    device = "cuda:0"
    metadata = pickle.load(open(f"{ROOT}/train.pkl", "rb"))
    all_dv = generate_all_dv(embedder, metadata, erroment_num)
    res, res_, all_yt, all_ys = [], [], [], []
    tmp_res = None
    for i in range(num_valid):
        tmp_res = generate_result(
            VC,
            embedder,
            metadata,
            i,
            all_dv,
            args.is_adain,
            args.is_vq,
            args.is_againvc,
        )
        rc, rc_ = get_cos(tmp_res)
        res.append(rc)
        res_.append(rc_)
        y_true, y_score = get_ytrue_yscore(tmp_res)
        for ele in y_true:
            all_yt.append(ele)
        for ele in y_score:
            all_ys.append(ele)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)

    res, res_ = np.array(res), np.array(res_)
    with open(f"{args.pt_name}_deepspeaker_result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(tpr)
        writer.writerow(fpr)
        writer.writerow(thresholds)
        writer.writerow([auc(fpr, tpr), res.mean(), res.std(), res_.mean(), res_.std()])
