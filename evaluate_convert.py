import pickle
import csv
import argparse
import torch
import importlib
import numpy as np
import json
from sklearn import metrics
from matplotlib import pyplot as plt
from util.evaluate import Evaluator
from factory.LstmDV import LstmDV
from factory.MetaDV import MetaDV
from sklearn.metrics import auc


json_config = json.load(open("config.json"))


class EConfig:
    def __init__(self, data_dir, embedder):
        self.root = data_dir
        self.embedder = embedder
        self.device = "cuda:0"
        # from config.json
        self.num_speaker = json_config["eval_speaker"]
        self.batch_size = json_config["batch_size"]
        self.erroment_num = json_config["erroment_num"]
        self.max_uttr_idx = json_config["num_uttr"] - 1
        self.len_crop = json_config["len_crop"]
        self.pick_speaker = json_config["pick_speaker"]
        self.metadata = pickle.load(open(f"{self.root}/train.pkl", "rb"))


if __name__ == "__main__":
    dim_neck = json_config["dim_neck"]
    dim_emb = json_config["dim_emb"]
    dim_pre = json_config["dim_pre"]
    freq = json_config["freq"]
    num_valid = json_config["num_valid"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="train_spmel_vctk80", required=False)
    parser.add_argument("--is_adain", default=False)
    parser.add_argument("--is_vq", default=False)
    parser.add_argument("--is_againvc", default=False)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pt_name", required=True)

    args = parser.parse_args()
    # modify embedder here
    C = LstmDV(913).to("cuda:0")
    # C = MetaDV(80, 256).to("cuda:0")
    C.load_state_dict(
        torch.load("model/static/dv_en_openslr.pth", map_location="cuda:0")
    )
    econfig = EConfig(data_dir=args.root_dir, embedder=C)
    E = Evaluator(econfig)
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

    res, res_, all_yt, all_ys = [], [], [], []
    tmp_res = None

    for i in range(num_valid):
        tmp_res = E.generate_result(
            [VC], i + 2, bool(args.is_adain), bool(args.is_vq), bool(args.is_againvc)
        )[0]
        rc, rc_ = E.get_cos(tmp_res)
        res.append(rc)
        res_.append(rc_)
        y_true, y_score = E.get_ytrue_yscore(tmp_res)
        for ele in y_true:
            all_yt.append(ele)
        for ele in y_score:
            all_ys.append(ele)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)

    plt.plot(fpr, tpr)
    plt.savefig(f"{args.pt_name}.png")

    res, res_ = np.array(res), np.array(res_)
    with open(f"{args.pt_name}_result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(tpr)
        writer.writerow(fpr)
        writer.writerow(thresholds)
        writer.writerow([auc(fpr, tpr), res.mean(), res.std(), res_.mean(), res_.std()])
