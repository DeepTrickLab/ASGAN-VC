import argparse
import json
import os
import pickle
import torch
import importlib
from util.evaluate import Evaluator
from torch.backends import cudnn
from util.data_loader import get_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--save_model_name", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--use_config", default="neck32", required=False)
    parser.add_argument("--device", default="cuda:0", required=False)
    parser.add_argument("--is_adain", default=False, required=False)
    parser.add_argument(
        "--method",
        default="Original",
        required=False,
        help="Set what training method you want to use",
    )
    parser.add_argument(
        "--is_validate",
        default=False,
        required=False,
        help="Set True if you want to validate when training",
    )
    args = parser.parse_args()
    json_config = json.load(open(f"config/{args.use_config}.json"))

    class Config:
        def __init__(
            self, model_name, data_dir, device, isadain, evaluator,
        ):
            # dynamic config
            self.model_name = model_name
            self.device = device
            self.data_dir = data_dir
            self.isadain = isadain
            self.evaluator = evaluator
            # static from config.json
            self.num_iters = json_config["num_iters"]
            self.validation_step = json_config["validation_step"]
            self.scheduler_step = json_config["scheduler_step"]
            self.num_speaker = json_config["num_speaker"]
            self.n_critic = json_config["n_critic"]
            self.batch_size = json_config["batch_size"]
            self.len_crop = json_config["len_crop"]
            self.lambda_cd = json_config["lambda_cd"]
            self.lambda_cls = json_config["lambda_cls"]
            self.lambda_dis = json_config["lambda_dis"]
            self.dim_neck = json_config["dim_neck"]
            self.dim_emb = json_config["dim_emb"]
            self.dim_pre = json_config["dim_pre"]
            self.freq = json_config["freq"]
            self.num_valid = json_config["num_valid"]
            self.log_step = json_config["log_step"]
            # GAN embedder, this is the pre-trained speaker embedding model
            print(
                f"Load Pretrained Embedder {json_config['GAN_embedder']} --- from --- {json_config['GAN_embedder_path']}"
            )
            self.GAN_embedder = getattr(
                importlib.import_module(f"factory.{json_config['GAN_embedder']}"),
                json_config["GAN_embedder"],
            )(80)
            # (json_config["num_speaker"])
            self.GAN_embedder.load_state_dict(
                torch.load(json_config["GAN_embedder_path"], map_location=self.device)
            )
            self.pretrained_step = int(self.num_iters / 10)

    if args.is_validate:

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
                self.embedder = getattr(
                    importlib.import_module(f"factory.{json_config['embedder']}"),
                    json_config["embedder"],
                )(json_config["embedder_speaker_num"])
                self.embedder.load_state_dict(
                    torch.load(json_config["embedder_path"], map_location=self.device)
                )
                self.pick_speaker = sorted(
                    next(iter(os.walk(self.root)))[1][: self.num_speaker]
                )
                self.metadata = pickle.load(open(f"{self.root}/train.pkl", "rb"))

        econfig = EConfig(data_dir=args.data_dir, device=args.device,)
        config = Config(
            model_name=args.model_name,
            data_dir=args.data_dir,
            device=args.device,
            isadain=bool(args.is_adain),
            evaluator=Evaluator(econfig),
        )
    else:
        config = Config(
            model_name=args.model_name,
            data_dir=args.data_dir,
            device=args.device,
            isadain=bool(args.is_adain),
            evaluator=None,
        )

    # 加速 conv，conv 的輸入 size 不會變的話開這個會比較快
    cudnn.benchmark = True
    # Data loader.
    vcc_loader = get_loader(
        config.data_dir,
        batch_size=config.batch_size,
        dim_neck=config.dim_neck,
        len_crop=config.len_crop,
    )

    try:
        solver = getattr(importlib.import_module(f"trainer.{args.method}"), "Solver")(
            vcc_loader, config
        )
        print(f"Using training Method --- {args.method}")
    except:
        print("Not a implemented Method, check your trainer!")
        raise NotImplementedError

    solver.train()
    torch.save(solver.VC.state_dict(), f"{args.save_model_name}.pt")
