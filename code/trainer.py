# trainer.py
import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data_utils import mae_mse
from losses import loss_recon_stage1, loss_align_stage2, total_loss_stage3
from mtfig import fit_linear_trend_params


# =========================
# ckpt helpers
# =========================
def save_ckpt(path, model, optim=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"model": model.state_dict()}
    if optim is not None:
        obj["optim"] = optim.state_dict()
    if extra is not None:
        obj["extra"] = extra
    torch.save(obj, path)


def load_ckpt(path, model, optim=None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"], strict=False)
    if optim is not None and "optim" in obj:
        optim.load_state_dict(obj["optim"])
    return obj.get("extra", None)


class EarlyStopper:
    def __init__(self, patience: int = 10):
        self.patience = int(patience)
        self.best = float("inf")
        self.bad = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - 1e-6:
            self.best = val_loss
            self.bad = 0
            return True
        self.bad += 1
        return False

    def should_stop(self) -> bool:
        return self.bad >= self.patience


def build_warmup_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.05
):
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _get_amp(cfg):
    use_amp = bool(getattr(cfg, "use_amp", True))
    device = str(getattr(cfg, "device", "cuda"))
    enabled = use_amp and ("cuda" in device)
    scaler = torch.amp.GradScaler("cuda") if enabled else None
    return enabled, scaler


def _build_param_groups(model: nn.Module, lr_head: float, lr_gpt: float, weight_decay: float):
    """
    分组学习率:
      - GPT2 参数用更小 lr_gpt
      - 其它模块用 lr_head
    LayerNorm/bias 不做 weight decay
    """
    decay, no_decay = [], []
    gpt_decay, gpt_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_gpt = name.startswith("gpt2.")
        is_no_decay = (
            name.endswith(".bias")
            or "ln" in name.lower()
            or "layernorm" in name.lower()
            or "norm" in name.lower()
        )

        if is_gpt:
            (gpt_no_decay if is_no_decay else gpt_decay).append(p)
        else:
            (no_decay if is_no_decay else decay).append(p)

    groups = []
    if len(decay) > 0:
        groups.append({"params": decay, "lr": lr_head, "weight_decay": weight_decay})
    if len(no_decay) > 0:
        groups.append({"params": no_decay, "lr": lr_head, "weight_decay": 0.0})
    if len(gpt_decay) > 0:
        groups.append({"params": gpt_decay, "lr": lr_gpt, "weight_decay": weight_decay})
    if len(gpt_no_decay) > 0:
        groups.append({"params": gpt_no_decay, "lr": lr_gpt, "weight_decay": 0.0})
    return groups


def _make_optimizer(model: nn.Module, cfg):
    lr_head = float(getattr(cfg, "lr", 1e-4))
    lr_gpt = float(getattr(cfg, "lr_gpt", max(1e-6, lr_head * 0.2)))
    wd = float(getattr(cfg, "weight_decay", 1e-3))
    groups = _build_param_groups(model, lr_head=lr_head, lr_gpt=lr_gpt, weight_decay=wd)
    return AdamW(groups)


def _build_time_weights(horizon: int, cfg, device, dtype):
    """Timestep weighting to slow error growth for long horizons.
    w(t) = 1 + lambda * (t/(H-1))^p, normalized to mean=1.
    """
    if not bool(getattr(cfg, "use_time_weight", True)):
        return None
    lam = float(getattr(cfg, "time_weight_lambda", 0.6))
    p = float(getattr(cfg, "time_weight_power", 1.5))
    if horizon <= 1 or lam <= 0:
        return None
    t = torch.linspace(0.0, 1.0, steps=int(horizon), device=device, dtype=dtype).view(1, -1, 1)
    w = 1.0 + lam * (t ** p)
    w = w / w.mean().clamp_min(1e-12)
    return w


def _amp_step(optim, scheduler, scaler, cfg, model):
    """
    AMP 下安全 step：
    - 先 scaler.step(optim) 再 scaler.update()
    - 如果发生 overflow 导致 step 被跳过，则不要 scheduler.step()
    """
    did_step = True
    if scaler is not None:
        scale_before = float(scaler.get_scale())
        scaler.step(optim)      # 可能因为 inf/NaN 而跳过
        scaler.update()
        scale_after = float(scaler.get_scale())
        # overflow 时 scale 会下降，意味着这一步 optimizer.step 被跳过
        did_step = (scale_after >= scale_before)
    else:
        optim.step()

    if scheduler is not None and did_step:
        scheduler.step()


@torch.no_grad()
def evaluate(model, loader, device, horizon: int):
    model.eval()
    maes_all, mses_all = [], []
    maes_inv_all, mses_inv_all = [], []
    scaler = getattr(loader.dataset, "scaler", None)
    mean = std = None
    if scaler is not None:
        mean = torch.as_tensor(scaler.mean, device=device, dtype=torch.float32)
        std = torch.as_tensor(scaler.std, device=device, dtype=torch.float32)

    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        y_hat, _ = model(x, horizon)

        mae_a, mse_a = mae_mse(y_hat, y)
        maes_all.append(mae_a)
        mses_all.append(mse_a)

        if scaler is not None:
            mean_ = mean.to(dtype=y.dtype)
            std_ = std.to(dtype=y.dtype)
            y_inv = y * std_ + mean_
            y_hat_inv = y_hat * std_ + mean_
            mae_i, mse_i = mae_mse(y_hat_inv, y_inv)
            maes_inv_all.append(mae_i)
            mses_inv_all.append(mse_i)

    mae_all = float(sum(maes_all) / len(maes_all))
    mse_all = float(sum(mses_all) / len(mses_all))

    if len(maes_inv_all) > 0:
        mae_inv_all = float(sum(maes_inv_all) / len(maes_inv_all))
        mse_inv_all = float(sum(mses_inv_all) / len(mses_inv_all))
        print(f"[Test-inv][ALL][H={horizon}] MAE={mae_inv_all:.6f}, MSE={mse_inv_all:.6f}")

    print(f"[Test][ALL][H={horizon}] MAE={mae_all:.6f}, MSE={mse_all:.6f}")
    return mae_all, mse_all


def train_stage1(model, train_loader, val_loader, cfg, horizon: int, ckpt_path: str):
    """
    Stage1: 局部重建（recon_len），避免重建整个 lookback。
    """
    device = cfg.device
    optim = _make_optimizer(model, cfg)

    use_sched = bool(getattr(cfg, "use_scheduler", True))
    warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.1))
    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.05))
    total_steps = int(cfg.stage1_epochs * max(1, len(train_loader)))
    scheduler = build_warmup_cosine_scheduler(optim, total_steps, warmup_ratio, min_lr_ratio) if use_sched else None

    amp_enabled, scaler = _get_amp(cfg)
    stopper = EarlyStopper(cfg.early_stop_patience)
    best_path = ckpt_path.replace(".pt", "_best.pt")

    recon_len = int(getattr(cfg, "stage1_recon_len", 96))
    recon_len = max(16, min(recon_len, int(getattr(cfg, "lookback", 512))))

    for epoch in range(cfg.stage1_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"[Stage1][H={horizon}] {epoch+1}/{cfg.stage1_epochs}", leave=False):
            x = batch[0].to(device, non_blocking=True)
            x_tgt = x[:, -recon_len:, :]

            optim.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    x_hat, _ = model(x, recon_len)
                    loss = loss_recon_stage1(x_hat, x_tgt)
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                _amp_step(optim, scheduler, scaler, cfg, model)
            else:
                x_hat, _ = model(x, recon_len)
                loss = loss_recon_stage1(x_hat, x_tgt)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

        # val
        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device, non_blocking=True)
                x_tgt = x[:, -recon_len:, :]
                x_hat, _ = model(x, recon_len)
                vlosses.append(loss_recon_stage1(x_hat, x_tgt).item())
        vloss = float(sum(vlosses) / len(vlosses))

        if stopper.step(vloss):
            save_ckpt(best_path, model, optim, {"stage": 1, "epoch": epoch, "val_loss": vloss})
        if stopper.should_stop():
            break

    if os.path.exists(best_path):
        load_ckpt(best_path, model, map_location=device)
    else:
        print(f"[Warning] best ckpt not found: {best_path}, using last model.")
    return best_path


def train_stage2(model, train_loader, val_loader, cfg, horizon: int, ckpt_path: str):
    device = cfg.device
    optim = _make_optimizer(model, cfg)

    use_sched = bool(getattr(cfg, "use_scheduler", True))
    warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.1))
    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.05))
    total_steps = int(cfg.stage2_epochs * max(1, len(train_loader)))
    scheduler = build_warmup_cosine_scheduler(optim, total_steps, warmup_ratio, min_lr_ratio) if use_sched else None

    amp_enabled, scaler = _get_amp(cfg)
    stopper = EarlyStopper(cfg.early_stop_patience)
    best_path = ckpt_path.replace(".pt", "_best.pt")

    for epoch in range(cfg.stage2_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"[Stage2][H={horizon}] {epoch+1}/{cfg.stage2_epochs}", leave=False):
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    _, aux = model(x, horizon)
                    with torch.no_grad():
                        k_true, b_true, s_true, _ = fit_linear_trend_params(y)
                    loss = loss_align_stage2(aux["k_hat"], aux["b_hat"], aux["s_hat"], k_true, b_true, s_true)
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                _amp_step(optim, scheduler, scaler, cfg, model)
            else:
                _, aux = model(x, horizon)
                with torch.no_grad():
                    k_true, b_true, s_true, _ = fit_linear_trend_params(y)
                loss = loss_align_stage2(aux["k_hat"], aux["b_hat"], aux["s_hat"], k_true, b_true, s_true)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

        # val
        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                _, aux = model(x, horizon)
                with torch.no_grad():
                    k_true, b_true, s_true, _ = fit_linear_trend_params(y)
                vlosses.append(loss_align_stage2(aux["k_hat"], aux["b_hat"], aux["s_hat"], k_true, b_true, s_true).item())
        vloss = float(sum(vlosses) / len(vlosses))

        if stopper.step(vloss):
            save_ckpt(best_path, model, optim, {"stage": 2, "epoch": epoch, "val_loss": vloss})
        if stopper.should_stop():
            break

    load_ckpt(best_path, model, map_location=device)
    return best_path


def train_stage3(model, train_loader, val_loader, cfg, horizon: int, ckpt_path: str):
    device = cfg.device
    optim = _make_optimizer(model, cfg)

    use_sched = bool(getattr(cfg, "use_scheduler", True))
    warmup_ratio = float(getattr(cfg, "warmup_ratio", 0.1))
    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.05))
    total_steps = int(cfg.max_epochs * max(1, len(train_loader)))
    scheduler = build_warmup_cosine_scheduler(optim, total_steps, warmup_ratio, min_lr_ratio) if use_sched else None

    amp_enabled, scaler = _get_amp(cfg)
    stopper = EarlyStopper(cfg.early_stop_patience)
    best_path = ckpt_path.replace(".pt", "_best.pt")

    # Precompute horizon-aware weights once
    tw = _build_time_weights(horizon, cfg, device=device, dtype=torch.float32)
    ms_scales = getattr(cfg, "ms_scales", [2, 4, 8])
    ms_weight = float(getattr(cfg, "ms_weight", 0.0))
    pred_loss = str(getattr(cfg, "pred_loss", "mix"))
    smoothl1_beta = float(getattr(cfg, "smoothl1_beta", 0.5))

    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"[Stage3][H={horizon}] {epoch+1}/{cfg.max_epochs}", leave=False):
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    y_hat, aux = model(x, horizon)
                    if "mix_lambda" not in aux and hasattr(cfg, "mix_lambda"):
                        aux["mix_lambda"] = float(getattr(cfg, "mix_lambda", 0.7))
                    aux.setdefault("pred_loss", pred_loss)
                    aux.setdefault("smoothl1_beta", smoothl1_beta)
                    if tw is not None:
                        aux["time_weights"] = tw.to(device=device, dtype=y_hat.dtype)
                    if ms_weight > 0:
                        aux["ms_scales"] = list(ms_scales)
                        aux["ms_weight"] = ms_weight
                    loss, _ = total_loss_stage3(
                        y_hat, y, aux,
                        alpha_mtfig=cfg.alpha_mtfig,
                        beta_hyper=cfg.beta_hyper,
                        gamma_consistency=cfg.gamma_consistency
                    )
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                _amp_step(optim, scheduler, scaler, cfg, model)
            else:
                y_hat, aux = model(x, horizon)
                if "mix_lambda" not in aux and hasattr(cfg, "mix_lambda"):
                    aux["mix_lambda"] = float(getattr(cfg, "mix_lambda", 0.7))
                aux.setdefault("pred_loss", pred_loss)
                aux.setdefault("smoothl1_beta", smoothl1_beta)
                if tw is not None:
                    aux["time_weights"] = tw.to(device=device, dtype=y_hat.dtype)
                if ms_weight > 0:
                    aux["ms_scales"] = list(ms_scales)
                    aux["ms_weight"] = ms_weight
                loss, _ = total_loss_stage3(
                    y_hat, y, aux,
                    alpha_mtfig=cfg.alpha_mtfig,
                    beta_hyper=cfg.beta_hyper,
                    gamma_consistency=cfg.gamma_consistency
                )
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                if scheduler is not None:
                    scheduler.step()

        # val
        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                y_hat, aux = model(x, horizon)
                if "mix_lambda" not in aux and hasattr(cfg, "mix_lambda"):
                    aux["mix_lambda"] = float(getattr(cfg, "mix_lambda", 0.7))
                aux.setdefault("pred_loss", pred_loss)
                aux.setdefault("smoothl1_beta", smoothl1_beta)
                if tw is not None:
                    aux["time_weights"] = tw.to(device=device, dtype=y_hat.dtype)
                if ms_weight > 0:
                    aux["ms_scales"] = list(ms_scales)
                    aux["ms_weight"] = ms_weight
                vloss, _ = total_loss_stage3(
                    y_hat, y, aux,
                    alpha_mtfig=cfg.alpha_mtfig,
                    beta_hyper=cfg.beta_hyper,
                    gamma_consistency=cfg.gamma_consistency
                )
                vlosses.append(float(vloss.item()))
        vloss = float(sum(vlosses) / len(vlosses))

        if stopper.step(vloss):
            save_ckpt(best_path, model, optim, {"stage": 3, "epoch": epoch, "val_loss": vloss})
        if stopper.should_stop():
            break

    load_ckpt(best_path, model, map_location=device)
    return best_path