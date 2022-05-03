# -*- coding: utf-8 -*-
import csv
import datetime
import importlib
import os
import time

import torch
import torch.nn.functional as F


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

                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )
                for tag in self.log_keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

            if (i + 1) % self.validation_step == 0 and self.evaluator != None:
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

        if self.evaluator != None:
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
