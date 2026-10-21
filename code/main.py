import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import json
import gc
import copy
import torch

from config import ExpConfig
from data_utils import set_seed, build_loaders_for_dataset
from model_ghef import GHEF
from trainer import train_stage1, train_stage2, train_stage3, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/data/xxx/GHEF/datasets/")
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpt2_name", type=str, default="/data/xxx/GHEF/gpt2_offline")

    p.add_argument("--datasets", type=str, nargs="*", default=[
        "ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic"
    ])
    p.add_argument("--horizons", type=int, nargs="*", default=[96, 192, 336, 720])
    p.add_argument("--lookback", type=int, default=512)

    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=20)
    p.add_argument("--no_stage1", action="store_true")
    p.add_argument("--no_stage2", action="store_true")
    p.add_argument("--no_stage3", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def apply_dataset_overrides(base_cfg: ExpConfig, dataset_name: str) -> ExpConfig:
    cfg = copy.deepcopy(base_cfg)
    overrides = (getattr(cfg, "dataset_special_overrides", None) or {}).get(dataset_name, None)
    if overrides is None:
        return cfg
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def main():
    args = parse_args()
    cfg = ExpConfig()
    cfg.data_root = args.data_root
    cfg.out_dir = args.out_dir
    cfg.device = args.device
    cfg.gpt2_name = args.gpt2_name
    cfg.datasets = args.datasets
    cfg.horizons = args.horizons
    cfg.lookback = args.lookback

    cfg.topk_hyper = args.topk
    cfg.max_epochs = args.epochs
    cfg.early_stop_patience = args.patience
    cfg.batch_size = args.batch
    cfg.lr = args.lr
    cfg.stage1_epochs = args.stage1_epochs
    cfg.stage2_epochs = args.stage2_epochs
    cfg.run_stage1 = not args.no_stage1
    cfg.run_stage2 = not args.no_stage2
    cfg.run_stage3 = not args.no_stage3
    cfg.num_workers = args.num_workers
    cfg.use_amp = not args.no_amp

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    all_results = {}

    for ds in cfg.datasets:
        ds_cfg = apply_dataset_overrides(cfg, ds)

        print(f"\n================ Dataset: {ds} ================")
        if ds_cfg is not cfg:
            print(
                f"[Dataset Override] {ds}: lookback={ds_cfg.lookback}, "
                f"n_granules={ds_cfg.n_granules}, "
                f"use_metadata_prompt={ds_cfg.use_metadata_prompt}, "
                f"max_prompt_tokens={ds_cfg.max_prompt_tokens}"
            )

        csv_path = os.path.join(ds_cfg.data_root, ds, f"{ds}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find csv: {csv_path}")

        ds_results = {}
        maes, mses = [], []

        for H in ds_cfg.horizons:
            print(f"\n---- Horizon: {H} ----")
            train_loader, val_loader, test_loader, scaler, D = build_loaders_for_dataset(
                csv_path=csv_path,
                lookback=ds_cfg.lookback,
                horizon=H,
                batch_size=ds_cfg.batch_size,
                dataset_name=ds,
                split_ratio=ds_cfg.split_ratio,
                zscore=ds_cfg.zscore_per_channel,
                drop_first_column_if_non_numeric=ds_cfg.drop_first_column_if_non_numeric,
                num_workers=ds_cfg.num_workers,
                pin_memory=ds_cfg.pin_memory,
                persistent_workers=getattr(ds_cfg, "persistent_workers", True),
                prefetch_factor=getattr(ds_cfg, "prefetch_factor", 2),
                drop_last=getattr(ds_cfg, "drop_last", True),
            )

            model = GHEF(D=D, cfg=ds_cfg).to(ds_cfg.device)
            model.set_dataset_name(ds)

            ckpt_dir = os.path.join(ds_cfg.out_dir, ds, f"H{H}")
            os.makedirs(ckpt_dir, exist_ok=True)

            if ds_cfg.run_stage1:
                s1_path = os.path.join(ckpt_dir, "stage1.pt")
                best1 = train_stage1(model, train_loader, val_loader, ds_cfg, horizon=H, ckpt_path=s1_path)
                print(f"[Stage1] best: {best1}")

            if ds_cfg.run_stage2:
                s2_path = os.path.join(ckpt_dir, "stage2.pt")
                best2 = train_stage2(model, train_loader, val_loader, ds_cfg, horizon=H, ckpt_path=s2_path)
                print(f"[Stage2] best: {best2}")

            if ds_cfg.run_stage3:
                s3_path = os.path.join(ckpt_dir, "stage3.pt")
                best3 = train_stage3(model, train_loader, val_loader, ds_cfg, horizon=H, ckpt_path=s3_path)
                print(f"[Stage3] best: {best3}")

            mae, mse = evaluate(model, test_loader, ds_cfg.device, horizon=H)
            print(f"[Test][{ds}][H={H}] MAE={mae:.6f}, MSE={mse:.6f}")
            ds_results[H] = {"MAE": mae, "MSE": mse}
            maes.append(mae)
            mses.append(mse)

            try:
                del train_loader, val_loader, test_loader, scaler
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ds_results["avg_4h"] = {
            "MAE": float(sum(maes) / len(maes)),
            "MSE": float(sum(mses) / len(mses)),
        }
        print(
            f"\n[AVG 4 horizons][{ds}] "
            f"MAE={ds_results['avg_4h']['MAE']:.6f}, MSE={ds_results['avg_4h']['MSE']:.6f}"
        )

        all_results[ds] = ds_results
        with open(os.path.join(ds_cfg.out_dir, f"results_{ds}.json"), "w", encoding="utf-8") as f:
            json.dump(ds_results, f, indent=2)

    with open(os.path.join(cfg.out_dir, "results_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll done. Saved to:", cfg.out_dir)


if __name__ == "__main__":
    main()
