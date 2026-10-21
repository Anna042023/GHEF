# data_utils.py
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_csv_multivariate(path: str, drop_first_column_if_non_numeric: bool = True) -> np.ndarray:
    df = pd.read_csv(path)
    if drop_first_column_if_non_numeric and df.shape[1] >= 2:
        if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            df = df.iloc[:, 1:]
    df = df.select_dtypes(include=[np.number])
    data = df.to_numpy(dtype=np.float32)
    if data.ndim != 2 or data.shape[0] < 10 or data.shape[1] < 1:
        raise ValueError(f"Bad csv numeric matrix: {path}, shape={data.shape}")
    return data  # [T, D]

def split_by_ratio(T: int, ratio=(0.7, 0.1, 0.2)) -> Tuple[slice, slice, slice]:
    n_train = int(T * ratio[0])
    n_val = int(T * ratio[1])
    s_train = slice(0, n_train)
    s_val = slice(n_train, n_train + n_val)
    s_test = slice(n_train + n_val, T)
    return s_train, s_val, s_test

def split_ett_fixed_borders(
    T: int,
    lookback: int,
    dataset_name: str
) -> Tuple[slice, slice, slice]:
    """
    复刻你给的 ETT 划分逻辑（只做 border 切分，不引入 label_len）：
      - ETTh: 12*30*24, 4*30*24, 4*30*24
      - ETTm: 12*30*24*4, 4*30*24*4, 4*30*24*4
    注意：你的代码对 val/test 的 border1 会减 seq_len，以保证窗口对齐。
    """
    name = (dataset_name or "").lower()
    if "ettm" in name:
        unit = 4
    else:
        unit = 1

    a = 12 * 30 * 24 * unit
    b = 4  * 30 * 24 * unit
    c = 4  * 30 * 24 * unit

    # 若 T 不足够长，就退回 ratio
    if T < (a + b + c):
        return split_by_ratio(T, ratio=(0.7,0.1,0.2))

    border1s = [0, a - lookback, a + b - lookback]
    border2s = [a, a + b, a + b + c]

    # 这里返回 slice，是给 loader 用的
    s_train = slice(border1s[0], border2s[0])
    s_val   = slice(border1s[1], border2s[1])
    s_test  = slice(border1s[2], border2s[2])
    return s_train, s_val, s_test

class ZScoreScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

class SafeWindowDataset(Dataset):
    """
    Lightweight sliding-window dataset.
    窗口长度在 __init__ 中已经保证合法，因此 __getitem__ 里不再做 pad，
    直接返回 view，减少 Python / NumPy 开销。
    """
    def __init__(self, series: np.ndarray, lookback: int, horizon: int):
        series = np.asarray(series, dtype=np.float32, order="C")
        self.series = torch.from_numpy(series)
        self.L = int(lookback)
        self.H = int(horizon)
        self.T, self.D = series.shape
        self.n = max(0, self.T - self.L - self.H + 1)
        if self.n <= 0:
            raise ValueError(f"Too short series: T={self.T}, L={self.L}, H={self.H}")
        self.scaler = None  # assigned externally

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        idx = int(idx)
        x = self.series[idx:idx + self.L]
        y = self.series[idx + self.L: idx + self.L + self.H]
        return x, y

def build_loaders_for_dataset(
    csv_path: str,
    lookback: int,
    horizon: int,
    batch_size: int,
    dataset_name: Optional[str] = None,
    split_ratio=(0.7, 0.1, 0.2),
    zscore=True,
    drop_first_column_if_non_numeric=True,
    num_workers=4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
):
    raw = read_csv_multivariate(csv_path, drop_first_column_if_non_numeric=drop_first_column_if_non_numeric)
    T, D = raw.shape

    name = (dataset_name or os.path.basename(os.path.dirname(csv_path))).lower()
    if name.startswith("etth") or name.startswith("ettm"):
        s_train, s_val, s_test = split_ett_fixed_borders(T, lookback=lookback, dataset_name=name)
    else:
        s_train, s_val, s_test = split_by_ratio(T, ratio=split_ratio)

    scaler = ZScoreScaler()
    if zscore:
        scaler.fit(raw[s_train])
        series = scaler.transform(raw).astype(np.float32, copy=False)
    else:
        series = raw.astype(np.float32, copy=False)

    train_ds = SafeWindowDataset(series[s_train], lookback, horizon)
    val_ds   = SafeWindowDataset(series[s_val], lookback, horizon)
    test_ds  = SafeWindowDataset(series[s_test], lookback, horizon)

    train_ds.scaler = scaler if zscore else None
    val_ds.scaler   = scaler if zscore else None
    test_ds.scaler  = scaler if zscore else None

    dl_kwargs = dict(
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and (int(num_workers) > 0),
    )
    if int(num_workers) > 0:
        dl_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=bool(drop_last),  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, **dl_kwargs)
    return train_loader, val_loader, test_loader, scaler, D

def mae_mse(pred: torch.Tensor, true: torch.Tensor):
    mae = (pred - true).abs().mean().item()
    mse = ((pred - true) ** 2).mean().item()
    return mae, mse

# ===== Compatibility helpers for precompute_granules.py =====
def load_dataset_csv(path: str, drop_first_column_if_non_numeric: bool = True) -> np.ndarray:
    return read_csv_multivariate(path, drop_first_column_if_non_numeric=drop_first_column_if_non_numeric)

def build_splits_indices(
    T: int,
    split_ratio=(0.7,0.1,0.2),
    lookback: int = 512,
    horizon: int = 1,
    dataset_name: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    给 precompute 用：返回“窗口起点索引”列表。
    ✅ 对 ETT 使用固定 border，与 build_loaders_for_dataset 完全一致的切分。
    """
    name = (dataset_name or "").lower()
    if name.startswith("etth") or name.startswith("ettm"):
        s_train, s_val, s_test = split_ett_fixed_borders(T, lookback=lookback, dataset_name=name)
    else:
        s_train, s_val, s_test = split_by_ratio(T, ratio=split_ratio)

    def indices_for_slice(s: slice):
        start = int(s.start); end = int(s.stop)
        n = max(0, (end - start) - lookback - horizon + 1)
        return list(range(start, start + n))

    return indices_for_slice(s_train), indices_for_slice(s_val), indices_for_slice(s_test)

class SlidingWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, indices, lookback: int, horizon: int):
        series = np.asarray(series, dtype=np.float32, order="C")
        self.series = torch.from_numpy(series)
        self.indices = list(indices)
        self.L = int(lookback)
        self.H = int(horizon)
        self.T, self.D = series.shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.series[idx: idx + self.L]
        y = self.series[idx + self.L: idx + self.L + self.H]
        return x.numpy(), y.numpy()
