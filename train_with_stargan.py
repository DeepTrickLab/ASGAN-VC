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

from factory.Discriminator import Discriminator
from factory.MetaDV import MetaDV
from util.data_loader import get_loader
from util.evaluate import Evaluator


class Solver(object):
    """
    AutoVC + StarGan
    """

    def __init__(self, vcc_loader, config):

        # Data loader
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.lambda_cls = config.lambda_cls
        self.lambda_dis = config.lambda_dis
        self.lambda_cycle = config.lambda_cycle
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.len_crop = config.len_crop
        self.n_critic = config.n_critic
        self.model_name = config.model_name
        self.num_speaker = config.num_speaker

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.pretrained_step = config.pretrained_step
        self.validation_step = config.validation_step
        self.num_valid = config.num_valid
        self.scheduler_step = config.scheduler_step
        self.isadain = config.isadain
        self.pretrained_embedder_path = "model/static/metadv_vctk80.pth"

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = config.device
        self.cosin_label = torch.ones(self.batch_size).to(self.device)
        self.evaluator = config.evaluator
        self.log_step = config.log_step
        self.log_keys = [
            "VC/loss",
            "C/loss",
            "D/loss",
            "Cycle/loss",
        ]

        # Build the model.
        self.build_model()

    def build_model(self):

        self.VC = getattr(
            importlib.import_module(f"factory.{self.model_name}"), self.model_name
        )(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        self.vc_optimizer = torch.optim.Adam(self.VC.parameters(), 0.0001)
        self.VC.to(self.device)

        # 拿來分類轉換後聲音的，算 speaker embedding 之間的 cos-similiarty 或 MSE
        self.C = MetaDV(self.num_speaker, 256)
        print(f"Load Pretrained Embedder from --- {self.pretrained_embedder_path}")
        self.C.load_state_dict(
            torch.load(self.pretrained_embedder_path, map_location=self.device)
        )
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), 0.0001)
        self.C.to(self.device)

        # 判斷聲音真假
        self.D = Discriminator(crop_len=self.len_crop, dim_neck=self.dim_neck)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), 0.0001)
        self.D.to(self.device)

        self.vc_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.vc_optimizer, gamma=0.85
        )
        self.c_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.c_optimizer, gamma=0.85
        )
        self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.d_optimizer, gamma=0.85
        )

    def reset_grad(self):
        self.vc_optimizer.zero_grad()
        self.c_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def discriminator_loss(self, real, fake):
        real_loss = torch.nn.BCELoss()(real, torch.ones_like(real))
        fake_loss = torch.nn.BCELoss()(fake, torch.zeros_like(fake))
        return real_loss + fake_loss

    def get_data(self):
        try:
            x, style = next(data_iter)
        except:
            data_iter = iter(self.vcc_loader)
            x, style = next(data_iter)
        x = x.to(self.device)
        style = style.to(self.device)
        return x, style

    def get_trans_mel(self, x_source, x_target, org_style, target_style):

        if self.isadain:
            _, target_feature = self.VC(x_target, target_style, None)
            _, x_trans, _ = self.VC(x_source, org_style, target_style, target_feature)
        else:
            _, x_trans, _ = self.VC(x_source, org_style, target_style)
        return x_trans

    def get_autovc_loss(
        self, x_source, x_identic, x_identic_psnt, code_real, code_reconst
    ):
        # Identity mapping loss
        vc_loss_id = F.mse_loss(x_source, x_identic.squeeze())
        vc_loss_id_psnt = F.mse_loss(x_source, x_identic_psnt.squeeze())
        # Code semantic loss.
        vc_loss_cd = F.l1_loss(code_real, code_reconst)
        return vc_loss_id + vc_loss_id_psnt + self.lambda_cd * vc_loss_cd

    def print_log(self, loss, i, start_time):
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
        for tag in self.log_keys:
            log += ", {}: {:.4f}".format(tag, loss[tag])
        print(log)

    def train(self):
        res_trans, res_trans_ = [], []
        print("Start training...")
        start_time = time.time()
        for i in range(self.num_iters):
            self.VC = self.VC.train()
            self.C = self.C.train()
            self.D = self.D.train()
            x_source, org_style = self.get_data()
            x_identic, x_identic_psnt, code_real = self.VC(
                x_source, org_style, org_style
            )
            if self.isadain:
                code_reconst, _ = self.VC(x_identic_psnt, org_style, None)
            else:
                code_reconst = self.VC(x_identic_psnt, org_style, None)

            # 更新 Classifier 跟 Discriminator
            if i % self.n_critic == 0 and i > self.pretrained_step:
                x_target, target_style = self.get_data()
                x_trans = self.get_trans_mel(
                    x_source, x_target, org_style, target_style
                )
                # Cycle Reconstruct, Target to Source domain
                x_reconstruct = self.get_trans_mel(
                    x_trans.squeeze(), x_source, target_style, org_style
                )

                # 抽出 style
                _, trans_style = self.C(x_trans.squeeze())
                _, reconstruct_style = self.C(x_reconstruct.squeeze())
                # Discriminator
                real_prob_trans = self.D(x_target)
                fake_prob_trans = self.D(x_trans.squeeze())
                real_prob_reconst = self.D(x_source)
                fake_prob_reconst = self.D(x_reconstruct.squeeze())

                # Classifier loss, Cosine Embedding Loss 低於 0.1 時代表轉換不錯
                c_loss = F.cosine_embedding_loss(
                    trans_style, target_style, self.cosin_label
                ) + F.cosine_embedding_loss(
                    reconstruct_style, org_style, self.cosin_label
                )
                # Discriminator loss
                d_loss = self.discriminator_loss(
                    real_prob_trans, fake_prob_trans
                ) + self.discriminator_loss(real_prob_reconst, fake_prob_reconst)

                self.reset_grad()
                c_loss.backward(retain_graph=True)
                self.c_optimizer.step()

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            if (i + 1) % self.n_critic == 0 and (i + 1) > self.pretrained_step:
                x_target, target_style = self.get_data()
                x_trans = self.get_trans_mel(
                    x_source, x_target, org_style, target_style
                )
                # Cycle Reconstruct, Target to Source domain
                x_reconstruct = self.get_trans_mel(
                    x_trans.squeeze(), x_source, target_style, org_style
                )

                # 抽出 style
                _, trans_style = self.C(x_trans.squeeze())
                _, reconstruct_style = self.C(x_reconstruct.squeeze())

                # x_target 是 Real Data
                real_prob_trans = self.D(x_target)
                fake_prob_trans = self.D(x_trans.squeeze())
                real_prob_reconst = self.D(x_source)
                fake_prob_reconst = self.D(x_reconstruct.squeeze())

                # AutoVC loss (未轉換的)
                autovc_loss = self.get_autovc_loss(
                    x_source, x_identic, x_identic_psnt, code_real, code_reconst
                )
                # Classifier loss, Cosine Embedding Loss 低於 0.1 時代表轉換不錯
                c_loss = F.cosine_embedding_loss(
                    trans_style, target_style, self.cosin_label
                ) + F.cosine_embedding_loss(
                    reconstruct_style, org_style, self.cosin_label
                )
                # Discriminator loss, 因為生成的 x_identic_psnt 沒有 Real Data 做比較，只能判斷真假
                d_loss = self.discriminator_loss(
                    real_prob_trans, fake_prob_trans
                ) + self.discriminator_loss(real_prob_reconst, fake_prob_reconst)

                # cycle loss Target to Source 要越像越好
                cycle_loss = F.mse_loss(x_source, x_reconstruct.squeeze())

                autovc_gan_loss = (
                    autovc_loss
                    + self.lambda_cls * c_loss
                    + self.lambda_dis * d_loss
                    + self.lambda_cycle * cycle_loss
                )
                self.reset_grad()
                autovc_gan_loss.backward()
                self.vc_optimizer.step()

            # 一般的 AutoVC
            else:
                # AutoVC loss
                autovc_loss = self.get_autovc_loss(
                    x_source, x_identic, x_identic_psnt, code_real, code_reconst
                )
                self.reset_grad()
                autovc_loss.backward()
                self.vc_optimizer.step()

            if (i + 1) == self.scheduler_step:
                self.vc_scheduler.step()
                self.c_scheduler.step()
                self.d_scheduler.step()

            if (i + 1) > self.pretrained_step and (i + 1) % self.log_step == 0:
                loss = {}
                loss["VC/loss"] = autovc_loss.item()
                loss["C/loss"] = c_loss.item()
                loss["D/loss"] = d_loss.item()
                loss["Cycle/loss"] = cycle_loss.item()
                self.print_log(loss, i, start_time)

            elif (i + 1) % self.log_step == 0:
                loss = {}
                loss["VC/loss"] = autovc_loss.item()
                loss["C/loss"] = 0.0
                loss["D/loss"] = 0.0
                loss["Cycle/loss"] = 0.0
                self.print_log(loss, i, start_time)

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

        with open(f"{self.model_name}_stargan_result.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                torch.arange(
                    self.validation_step,
                    self.num_iters + self.validation_step,
                    self.validation_step,
                ).numpy()
            )
            writer.writerow(res_trans)
            writer.writerow(res_trans_)


# Common JSON config
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
        self.lambda_cd = json_config["lambda_cd"]
        self.lambda_cls = json_config["lambda_cls"]
        self.lambda_dis = json_config["lambda_dis"]
        self.n_critic = json_config["n_critic"]
        self.batch_size = json_config["batch_size"]
        self.len_crop = json_config["len_crop"]
        self.dim_neck = json_config["dim_neck"]
        self.dim_emb = json_config["dim_emb"]
        self.dim_pre = json_config["dim_pre"]
        self.freq = json_config["freq"]
        self.num_valid = json_config["num_valid"]
        self.log_step = json_config["log_step"]
        # eval
        self.evaluator = evaluator
        # other
        self.pretrained_step = int(self.num_iters / 10)


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

    print(" --- Use Config  ---")
    print(f" Lambda cd ---  {config.lambda_cd}")
    print(f" Lambda cls --- {config.lambda_cls}")
    print(f" Lambda dis--- {config.lambda_dis}")
    print(f" Lambda cycle --- {config.lambda_cycle}")
    print(f" N critic --- {config.n_critic }")
    print(f" Use Adain --- {config.isadain }")
    print(f" VC Pretrained step  --- {config.pretrained_step  }")
    print(" ----------------------")

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

    with open(
        f"{args.model_name}_stargan_final_result.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(TPR)
        writer.writerow(FPR)
        writer.writerow([auc_res, res.mean(), res.std(), res_.mean(), res_.std()])
    """
