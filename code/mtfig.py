
# mtfig.py
import torch
import torch.nn as nn


class ReverseInstanceNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, d))
        self.beta  = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        sigma = (var + self.eps).sqrt()
        x_hat = (x - mu) / sigma
        x_tilde = x_hat * self.gamma + self.beta
        return x_tilde, (mu, sigma)


def fit_linear_trend_params(x: torch.Tensor):
    """
    Fit least-squares line per (B,D) on length L:
      x_t ≈ k * t + b, sigma = std(residual), T = L
    """
    B, L, D = x.shape
    t = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1)
    t_mean = t.mean(dim=1, keepdim=True)
    x_mean = x.mean(dim=1, keepdim=True)

    num = ((t - t_mean) * (x - x_mean)).sum(dim=1)                  # [B,D]
    den = ((t - t_mean) ** 2).sum(dim=1).view(1) + 1e-8             # scalar
    k = num / den
    b = x_mean.squeeze(1) - k * t_mean.squeeze(1).squeeze(-1)

    trend = k.unsqueeze(1) * t + b.unsqueeze(1)
    resid = x - trend
    sigma = resid.std(dim=1, unbiased=False) + 1e-6
    T = torch.full_like(k, float(L))
    return k, b, sigma, T


def _membership(resid: torch.Tensor, scale: torch.Tensor, kind: str, eps: float):
    """
    resid: [B,L,D], scale: [B,D]
    """
    s = scale.unsqueeze(1).clamp(min=eps)
    kind = (kind or "gaussian").lower()
    if kind == "laplace":
        return torch.exp(-resid.abs() / s)
    if kind == "cauchy":
        z = resid / s
        return 1.0 / (1.0 + z * z)
    return torch.exp(-0.5 * (resid / s) ** 2)


@torch.no_grad()
def _approx_l1_trend_breaks(x: torch.Tensor, G: int, min_gap: int = 8):
    """
    Fast approximation to ℓ1-trend filtering segmentation:
    pick (G-1) breakpoints at the largest aggregated 2nd-difference magnitudes,
    with non-max suppression (min_gap) to avoid clustered breaks.

    x: [B,L,D] (already RevIN-normalized in model)
    returns: breaks [B,D,G+1] with 0 and L included
    """
    B, L, D = x.shape
    if G <= 1 or L < 4:
        br = torch.zeros((B, D, 2), device=x.device, dtype=torch.long)
        br[..., 1] = L
        return br

    # second difference magnitude (curvature proxy)
    dd = x[:, 2:, :] - 2.0 * x[:, 1:-1, :] + x[:, :-2, :]           # [B,L-2,D]
    score = dd.abs()                                                # [B,L-2,D]
    # candidate indices correspond to t=1..L-2 (align to original)
    idx_base = torch.arange(1, L - 1, device=x.device).view(1, -1, 1)  # [1,L-2,1]

    # greedy select top with suppression, per (B,D)
    breaks = torch.zeros((B, D, G + 1), device=x.device, dtype=torch.long)
    breaks[:, :, 0] = 0
    breaks[:, :, -1] = L

    # mask out near boundaries
    valid = torch.ones_like(score, dtype=torch.bool)
    valid[:, :min_gap, :] = False
    valid[:, -min_gap:, :] = False

    chosen = torch.zeros((B, D, 0), device=x.device, dtype=torch.long)

    # iterative selection (G is small, so loop is cheap)
    used = torch.zeros_like(score, dtype=torch.bool)
    for k in range(G - 1):
        masked = score.clone()
        masked[~valid] = -1.0
        masked[used] = -1.0
        # best index along time
        best = masked.argmax(dim=1)                                  # [B,D] over L-2
        t = best                                                     # 0..L-3
        sel = (t + 1).clamp(min=1, max=L - 2)                        # to original time index
        breaks[:, :, 1 + k] = sel

        # suppress neighborhood
        for b in range(B):
            for d in range(D):
                s = int(sel[b, d].item()) - 1                        # map back to dd index
                lo = max(0, s - min_gap)
                hi = min(L - 2, s + min_gap + 1)
                used[b, lo:hi, d] = True

    # sort breakpoints
    breaks, _ = torch.sort(breaks, dim=-1)
    # ensure strictly increasing (fallback to uniform if duplicates)
    dif = breaks[:, :, 1:] - breaks[:, :, :-1]
    bad = (dif <= 0).any(dim=-1)
    if bad.any():
        # uniform for those cases
        seg_len = max(1, L // G)
        uni = torch.arange(0, G + 1, device=x.device).view(1, 1, -1) * seg_len
        uni[:, :, -1] = L
        breaks[bad] = uni.expand(bad.sum(), 1, G + 1).squeeze(1)
    return breaks


def build_mtfig_granules(x: torch.Tensor, cfg):
    """
    Output: granules [B,D,G,4] = (k,b,sigma,T), gmask [B,D,G]

    Paper uses ℓ1-trend filtering; here we provide a *fast approximation* that
    selects breakpoints based on curvature (2nd difference) and then fits lines
    inside each segment. This keeps runtime very low and keeps granules adaptive.

    Sigma is membership-weighted residual dispersion (Gaussian MT-FIG).
    """
    B, L, D = x.shape
    G = max(1, int(getattr(cfg, "n_granules", 4)))
    eps = float(getattr(cfg, "gaussian_eps", 1e-6))
    kind = str(getattr(cfg, "membership_dist", "cauchy")).lower()
    min_gap = int(getattr(cfg, "trend_min_gap", 8))

    if bool(getattr(cfg, "use_l1_trend_filter", True)) and G > 1:
        br = _approx_l1_trend_breaks(x, G=G, min_gap=min_gap)         # [B,D,G+1]
    else:
        # uniform
        seg_len = max(1, L // G)
        br = torch.arange(0, G + 1, device=x.device).view(1, 1, -1) * seg_len
        br[:, :, -1] = L
        br = br.expand(B, D, G + 1).contiguous()

    gran = torch.zeros((B, D, G, 4), device=x.device, dtype=x.dtype)
    gmask = torch.ones((B, D, G), device=x.device, dtype=torch.float32)

    # loop over G only (small), slicing is vectorized over B,D
    for g in range(G):
        s0 = br[:, :, g]
        e0 = br[:, :, g + 1]
        # gather segments via per-sample indexing (vectorized with masking)
        # build a (B,L,D) mask for current segment
        t = torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, D)
        seg_mask = (t >= s0.unsqueeze(1)) & (t < e0.unsqueeze(1))     # [B,L,D]
        # avoid empty
        lens = (e0 - s0).clamp(min=1).to(x.dtype)                     # [B,D]
        # compute t' in [0, len-1] for fitting
        t0 = (t.to(x.dtype) - s0.unsqueeze(1).to(x.dtype)).clamp(min=0)
        # compute means within segment
        w = seg_mask.to(x.dtype)
        wsum = w.sum(dim=1).clamp(min=1.0)                            # [B,D]
        t_mean = (w * t0).sum(dim=1) / wsum                           # [B,D]
        x_mean = (w * x).sum(dim=1) / wsum                            # [B,D]
        num = (w * (t0 - t_mean.unsqueeze(1)) * (x - x_mean.unsqueeze(1))).sum(dim=1)
        den = (w * (t0 - t_mean.unsqueeze(1)) ** 2).sum(dim=1) + 1e-8
        k = num / den                                                 # [B,D]
        b = x_mean - k * t_mean                                       # [B,D]

        # residuals
        trend = k.unsqueeze(1) * t0 + b.unsqueeze(1)
        resid = (x - trend) * w
        sigma_std = torch.sqrt((resid.pow(2).sum(dim=1) / wsum).clamp(min=eps)) + 1e-6

        mw = _membership(resid, sigma_std, kind=kind, eps=eps) * w
        mw_sum = mw.sum(dim=1).clamp(min=eps)
        var_w = (mw * resid.pow(2)).sum(dim=1) / mw_sum
        sigma = torch.sqrt(var_w + eps)

        T = lens
        gran[:, :, g, 0] = k
        gran[:, :, g, 1] = b
        gran[:, :, g, 2] = sigma
        gran[:, :, g, 3] = T

    return gran, gmask


class MTFIGGranuleEmbed(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, granules: torch.Tensor):
        B, D, G, _ = granules.shape
        tok = self.proj(granules)
        return tok.reshape(B, D * G, -1)
