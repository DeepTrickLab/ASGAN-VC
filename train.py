# -*- coding: utf-8 -*-
import argparse
import csv
import datetime
import importlib
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc
from torch.backends import cudnn

from util.data_loader import get_loader
from util.evaluate import Evaluator


class Solver(object):
    """
    最原始的 AutoVC 訓練方法
    """

    def __init__(self, vcc_loader, config):
        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.isadain = config.isadain
        self.model_name = config.model_name

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.validation_step = config.validation_step
        self.num_valid = config.num_valid
        self.scheduler_step = config.scheduler_step

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = config.device
        self.log_step = config.log_step
        self.evaluator = config.evaluator
        self.log_keys = ["VC/loss_id", "VC/loss_id_psnt", "VC/loss_cd"]

        # Build the model.
        self.build_model()

    def build_model(self):

        self.VC = getattr(
            importlib.import_module(f"factory.{self.model_name}"), self.model_name
        )(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)

        self.vc_optimizer = torch.optim.Adam(self.VC.parameters(), 0.0001)
        self.VC.to(self.device)

        self.vc_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.vc_optimizer, gamma=0.85
        )

    def reset_grad(self):
        self.vc_optimizer.zero_grad()

    def train(self):

        data_loader = self.vcc_loader

        res_trans, res_trans_ = [], []

        print("Start training...")
        start_time = time.time()
        for i in range(self.num_iters):
            self.VC = self.VC.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)

            x_real = x_real.to(self.device)
            emb_org = emb_org.to(self.device)

            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.VC(x_real, emb_org, emb_org)
            vc_loss_id = F.mse_loss(x_real, x_identic.squeeze())
            vc_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt.squeeze())

            # Code semantic loss.
            if self.isadain:
                code_reconst, _ = self.VC(x_identic_psnt, emb_org, None)
            else:
                code_reconst = self.VC(x_identic_psnt, emb_org, None)

            vc_loss_cd = F.l1_loss(code_real, code_reconst)
            # Backward and optimize.
            vc_loss = vc_loss_id + vc_loss_id_psnt + self.lambda_cd * vc_loss_cd
            self.reset_grad()
            vc_loss.backward()
            self.vc_optimizer.step()

            # Logging.
            loss = {}
            loss["VC/loss_id"] = vc_loss_id.item()
            loss["VC/loss_id_psnt"] = vc_loss_id_psnt.item()
            loss["VC/loss_cd"] = vc_loss_cd.item()

            if (i + 1) == self.scheduler_step:
                self.vc_scheduler.step()

            # =================================================================================== #
            #                               4. Print Traning Info                                 #
            # =================================================================================== #

            if (i + 1) % self.log_step == 0:
                """
                wandb.log(
                    {
                        "VC_LOSS_ID": vc_loss_id.item(),
                        "VC_LOSS_ID_PSNET": vc_loss_id_psnt.item(),
                        "VC_LOSS_CD": vc_loss_cd.item(),
                    }
                )
                """
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )
                for tag in self.log_keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

            if (i + 1) % self.validation_step == 0:
                print("--- Now Star Validation ---")
                tmp_trans, tmp_trans_ = 0.0, 0.0
                for i in range(self.num_valid):
                    cos_result = self.evaluator.generate_result(
                        [self.VC], i + 2, self.isadain
                    )
                    trans, trans_ = self.evaluator.get_cos(cos_result[0])
                    tmp_trans += trans
                    tmp_trans_ += trans_
                res_trans.append(trans / self.num_valid)
                res_trans_.append(trans_ / self.num_valid)

            if (i + 2) % self.log_step == 0:
                os.system("cls||clear")

        with open(f"{self.model_name}_result.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                torch.arange(
                    self.validation_step,
                    self.num_iters + self.validation_step,
                    self.validation_step,
                ).numpy(),
            )
            writer.writerow(res_trans)
            writer.writerow(res_trans_)


json_config = json.load(open("config.json"))


class EConfig:
    def __init__(self, data_dir, device):
        self.root = data_dir
        self.device = device
        # from config.json
        self.num_speaker = json_config["eval_speaker"]
        self.batch_size = json_config["batch_size"]
        self.erroment_num = json_config["erroment_num"]
        self.max_uttr_idx = json_config["num_uttr"] - 1
        self.len_crop = json_config["len_crop"]
        self.embedder = torch.load("model/static/dv_vctk80.pt").to(self.device)
        self.pick_speaker = sorted(
            next(iter(os.walk(self.root)))[1][: self.num_speaker]
        )
        self.metadata = pickle.load(open(f"{self.root}/train.pkl", "rb"))


class Config:
    def __init__(
        self, model_name, data_dir, device, isadain, evaluator,
    ):
        self.model_name = model_name
        self.device = device
        self.data_dir = data_dir
        self.isadain = isadain
        # from config.json
        self.num_iters = json_config["num_iters"]
        self.validation_step = json_config["validation_step"]
        self.scheduler_step = json_config["scheduler_step"]
        self.num_speaker = json_config["num_speaker"]
        self.n_critic = json_config["n_critic"]
        self.batch_size = json_config["batch_size"]
        self.len_crop = json_config["len_crop"]
        self.lambda_cd = json_config["lambda_cd"]
        self.dim_neck = json_config["dim_neck"]
        self.dim_emb = json_config["dim_emb"]
        self.dim_pre = json_config["dim_pre"]
        self.freq = json_config["freq"]
        self.num_valid = json_config["num_valid"]
        self.log_step = json_config["log_step"]
        # eval
        self.evaluator = evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--save_model_name", required=True)
    parser.add_argument("--data_dir", default="train_spmel_vctk80", required=False)
    parser.add_argument("--device", default="cuda:0", required=False)
    parser.add_argument("--is_adain", default=False, required=False)

    args = parser.parse_args()
    econfig = EConfig(data_dir=args.data_dir, device=args.device,)
    config = Config(
        model_name=args.model_name,
        data_dir=args.data_dir,
        device=args.device,
        isadain=bool(args.is_adain),
        evaluator=Evaluator(econfig),
    )

    ### Init Wandb
    """
    wandb.init(project=f'AutoVC {datetime.date.today().strftime("%b %d")}')
    wandb.run.name = args.model_name
    wandb.run.save()
    w_config = wandb.config
    w_config.len_crop = config.len_crop
    w_config.dim_neck = config.dim_neck
    w_config.dim_emb = config.dim_emb
    w_config.freq = config.freq
    w_config.batch_size = config.batch_size
    """

    # 加速 conv，conv 的輸入 size 不會變的話開這個會比較快
    cudnn.benchmark = True
    # Data loader.
    vcc_loader = get_loader(
        config.data_dir,
        batch_size=config.batch_size,
        dim_neck=config.dim_neck,
        len_crop=config.len_crop,
    )

    solver = Solver(vcc_loader, config)
    solver.train()
    torch.save(solver.VC.state_dict(), f"{args.save_model_name}.pt")

    """
    # After traning do evaluate
    E = Evaluator(econfig)
    res, res_ = [], []
    all_TP, all_FP, all_TN, all_FN = [], [], [], []
    tmp_res = None
    thresholds = np.arange(0, 1.01, 0.01)
    for i in range(5):
        tmp_res = E.generate_result([solver.VC], i + 2, bool(args.is_adain))[0]
        tmp_tp, tmp_fp, tmp_tn, tmp_fn = [], [], [], []
        for threshold in thresholds:
            TP, FP, TN, FN = E.get_confusion_matrix(tmp_res, threshold)
            tmp_tp.append(TP)
            tmp_fp.append(FP)
            tmp_tn.append(TN)
            tmp_fn.append(FN)
        all_TP.append(tmp_tp)
        all_FP.append(tmp_fp)
        all_TN.append(tmp_tn)
        all_FN.append(tmp_fn)
        rc, rc_ = E.get_cos(tmp_res)
        res.append(rc)
        res_.append(rc_)

    all_TP = np.array(all_TP).mean(axis=0)
    all_FP = np.array(all_FP).mean(axis=0)
    all_TN = np.array(all_TN).mean(axis=0)
    all_FN = np.array(all_FN).mean(axis=0)
    TPR = [tp / (tp + fn) for (tp, fn) in zip(all_TP, all_FN)]
    TPR.reverse()
    FPR = [fp / (fp + tn) for (fp, tn) in zip(all_FP, all_TN)]
    FPR.reverse()
    auc_res = auc(FPR, TPR)
    res = np.array(res)
    res_ = np.array(res_)

    with open(f"{args.model_name}_final_result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(TPR)
        writer.writerow(FPR)
        writer.writerow([auc_res, res.mean(), res.std(), res_.mean(), res_.std()])
    """
