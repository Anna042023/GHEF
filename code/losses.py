# losses.py
import torch
import torch.nn.functional as F


def loss_recon_stage1(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    # Eq.(17): reconstruction loss
    return F.mse_loss(x_hat, x_true)


def loss_align_stage2(k_hat, b_hat, s_hat, k_true, b_true, s_true) -> torch.Tensor:
    # Eq.(18): MT-FIG parameter alignment (use L1 for robustness)
    return F.l1_loss(k_hat, k_true) + F.l1_loss(b_hat, b_true) + F.l1_loss(s_hat, s_true)


def _pred_loss_default(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    默认用 SmoothL1，比纯 MSE 更接近 MAE 指标，训练更稳。
    你后面如果想切换，也可以在 aux 里塞 pred_loss/smoothl1_beta。
    """
    beta = 0.5
    return F.smooth_l1_loss(y_hat, y_true, beta=beta)


def total_loss_stage3(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    aux: dict,
    alpha_mtfig: float = 1.0,
    beta_hyper: float = 1e-3,
    gamma_consistency: float = 0.0,
):
    """
    Eq.(19) style:
      L = L_pred + alpha * L_mtfig + beta * L_hyper + gamma * L_consistency

    注意：trainer.py 现在就是按 (y_hat, y_true, aux, alpha_mtfig=..., beta_hyper=..., gamma_consistency=...) 来调用，
    所以这里必须匹配这个签名。
    """

    # ===== 1) prediction loss (horizon-stable) =====
    # Long horizons: uniform loss is dominated by early steps (small error),
    # so the model under-optimizes far steps -> MAE/MSE grows too fast.
    # We support optional timestep weights + multi-scale low-frequency supervision.
    pred_loss_type = aux.get("pred_loss", "smoothl1")
    w = aux.get("time_weights", None)  # [1,H,1] or [B,H,1]

    def _reduce(elem: torch.Tensor):
        # elem: [B,H,D]
        if w is None:
            return elem.mean()
        ww = w.to(device=elem.device, dtype=elem.dtype)
        if ww.dim() == 3 and ww.shape[-1] == 1:
            ww = ww.expand(elem.shape[0], elem.shape[1], elem.shape[2])
        denom = ww.sum().clamp_min(1e-12)
        return (elem * ww).sum() / denom

    if pred_loss_type == "mse":
        L_pred = _reduce((y_hat - y_true) ** 2)
    elif pred_loss_type == "mix":
        lam = float(aux.get("mix_lambda", 0.7))
        lam = max(0.0, min(1.0, lam))
        L_pred = lam * _reduce((y_hat - y_true).abs()) + (1.0 - lam) * _reduce((y_hat - y_true) ** 2)
    elif pred_loss_type == "mae" or pred_loss_type == "l1":
        L_pred = _reduce((y_hat - y_true).abs())
    else:
        beta = float(aux.get("smoothl1_beta", 0.5))
        L_pred = _reduce(F.smooth_l1_loss(y_hat, y_true, beta=beta, reduction="none"))

    # ---- multi-scale auxiliary loss (avg-pool along time) ----
    L_ms = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
    ms_scales = aux.get("ms_scales", None)  # e.g. [2,4,8]
    ms_weight = float(aux.get("ms_weight", 0.0) or 0.0)
    if ms_scales and ms_weight > 0:
        yh = y_hat.transpose(1, 2)  # [B,D,H]
        yt = y_true.transpose(1, 2)
        used = 0
        for s in ms_scales:
            s = int(s)
            if s <= 1 or yh.shape[-1] < s * 2:
                continue
            yh_p = F.avg_pool1d(yh, kernel_size=s, stride=s)
            yt_p = F.avg_pool1d(yt, kernel_size=s, stride=s)
            yh_p = yh_p.transpose(1, 2)
            yt_p = yt_p.transpose(1, 2)
            if pred_loss_type == "mse":
                L_ms = L_ms + F.mse_loss(yh_p, yt_p)
            elif pred_loss_type == "mix":
                lam = float(aux.get("mix_lambda", 0.7))
                lam = max(0.0, min(1.0, lam))
                L_ms = L_ms + (lam * F.l1_loss(yh_p, yt_p) + (1.0 - lam) * F.mse_loss(yh_p, yt_p))
            elif pred_loss_type == "mae" or pred_loss_type == "l1":
                L_ms = L_ms + F.l1_loss(yh_p, yt_p)
            else:
                beta = float(aux.get("smoothl1_beta", 0.5))
                L_ms = L_ms + F.smooth_l1_loss(yh_p, yt_p, beta=beta)
            used += 1
        if used > 0:
            L_ms = L_ms / float(used)

    # ---- tail-horizon loss (last third) ----
    tail_weight = float(aux.get("tail_weight", 0.0) or 0.0)
    L_tail = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
    if tail_weight > 0 and y_hat.shape[1] >= 3:
        tail_start = max(1, int(y_hat.shape[1] * 2 // 3))
        yh_tail = y_hat[:, tail_start:, :]
        yt_tail = y_true[:, tail_start:, :]
        if pred_loss_type == "mse":
            L_tail = F.mse_loss(yh_tail, yt_tail)
        elif pred_loss_type == "mix":
            lam = float(aux.get("mix_lambda", 0.7))
            lam = max(0.0, min(1.0, lam))
            L_tail = lam * F.l1_loss(yh_tail, yt_tail) + (1.0 - lam) * F.mse_loss(yh_tail, yt_tail)
        elif pred_loss_type == "mae" or pred_loss_type == "l1":
            L_tail = F.l1_loss(yh_tail, yt_tail)
        else:
            beta = float(aux.get("smoothl1_beta", 0.5))
            L_tail = F.smooth_l1_loss(yh_tail, yt_tail, beta=beta)

    # ---- first-difference loss: slow down long-horizon error explosion ----
    diff_weight = float(aux.get("diff_loss_weight", 0.0) or 0.0)
    L_diff = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
    if diff_weight > 0 and y_hat.shape[1] >= 2:
        dy_hat = y_hat[:, 1:, :] - y_hat[:, :-1, :]
        dy_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        L_diff = F.smooth_l1_loss(dy_hat, dy_true, beta=float(aux.get("smoothl1_beta", 0.5)))

    # ---- future trend-parameter alignment ----
    trend_align_weight = float(aux.get("trend_align_weight", 0.0) or 0.0)
    L_trend = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
    if trend_align_weight > 0:
        kf = aux.get("k_f_hat", None)
        bf = aux.get("b_f_hat", None)
        sf = aux.get("s_f_hat", None)
        kt = aux.get("k_f_true", None)
        bt = aux.get("b_f_true", None)
        st = aux.get("s_f_true", None)
        if kf is not None and bf is not None and sf is not None and kt is not None and bt is not None and st is not None:
            L_trend = F.l1_loss(kf, kt) + F.l1_loss(bf, bt) + 0.5 * F.l1_loss(sf, st)

    # ===== 2) MT-FIG regularization =====
    # 你的 trainer/model 如果有提供 L_mtfig，就用；没有就置 0
    L_mt = aux.get("L_mtfig", None)
    if L_mt is None:
        L_mt = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
    elif not torch.is_tensor(L_mt):
        L_mt = torch.tensor(float(L_mt), device=y_hat.device, dtype=y_hat.dtype)

    # ===== 3) Hypergraph regularization =====
    # A_hyper 一般是 [D,D]，做一个很轻的稀疏/强度正则即可
    A = aux.get("A_hyper", None)
    if A is None:
        L_hyper = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)
    else:
        # 防止过密导致过平滑：用平均绝对值做轻正则
        L_hyper = A.abs().mean()

    # ===== 4) consistency (默认不用) =====
    # 你现在 stage3 先跑通为主，这一项先置 0
    L_cons = torch.tensor(0.0, device=y_hat.device, dtype=y_hat.dtype)

    total = L_pred + ms_weight * L_ms + float(alpha_mtfig) * L_mt + float(beta_hyper) * L_hyper + float(gamma_consistency) * L_cons

    logs = {
        "L_pred": float(L_pred.detach().item()),
        "L_ms": float(L_ms.detach().item()),
        "L_mtfig": float(L_mt.detach().item()),
        "L_hyper": float(L_hyper.detach().item()),
        "L_cons": float(L_cons.detach().item()),
    }
    return total, logs