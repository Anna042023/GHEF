# precompute_granules.py
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config import ExpConfig
from data_utils import load_dataset_csv, build_splits_indices, SlidingWindowDataset
from mtfig import ReverseInstanceNorm, build_mtfig_granules

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True, help="where to save cached granules")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lookback", type=int, default=512)  # ✅对齐论文
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    cfg = ExpConfig()
    cfg.data_root = args.data_root
    cfg.lookback = args.lookback

    x_all = load_dataset_csv(
        os.path.join(args.data_root, args.dataset, f"{args.dataset}.csv"),
        drop_first_column_if_non_numeric=cfg.drop_first_column_if_non_numeric
    )
    T, D = x_all.shape

    # ✅ 用与训练一致的切分逻辑（ETT 固定 border / 其他 ratio）
    tr_idx, va_idx, te_idx = build_splits_indices(
        T,
        split_ratio=cfg.split_ratio,
        lookback=cfg.lookback,
        horizon=1,
        dataset_name=args.dataset
    )

    splits = {"train": tr_idx, "val": va_idx, "test": te_idx}

    # granules 构建基于 RIN/RevIN 后的 x（与你论文描述一致的方向）
    rin = ReverseInstanceNorm(D).to(args.device)
    rin.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    for split_name, indices in splits.items():
        ds = SlidingWindowDataset(x_all, indices, lookback=cfg.lookback, horizon=1)  # horizon 占位
        B = 64
        all_gran, all_gmask = [], []

        for i in tqdm(range(0, len(ds), B), desc=f"Precompute {args.dataset} {split_name}"):
            batch_x = []
            for j in range(i, min(i + B, len(ds))):
                x_win, _ = ds[j]
                batch_x.append(x_win)
            x = torch.from_numpy(np.stack(batch_x, 0)).to(args.device).float()  # [B,L,D]

            x_rin, _ = rin(x)
            gran, gmask = build_mtfig_granules(x_rin, cfg)

            all_gran.append(gran.detach().cpu())
            all_gmask.append(gmask.detach().cpu())

        all_gran = torch.cat(all_gran, dim=0)
        all_gmask = torch.cat(all_gmask, dim=0)

        save_path = os.path.join(args.out_dir, f"{args.dataset}_{split_name}_L{cfg.lookback}.pt")
        torch.save({"granules": all_gran, "gmask": all_gmask, "indices": indices}, save_path)
        print(f"[Saved] {save_path}  granules={tuple(all_gran.shape)} gmask={tuple(all_gmask.shape)}")

if __name__ == "__main__":
    main()