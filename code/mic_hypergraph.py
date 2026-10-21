# mic_hypergraph.py
import torch
import torch.nn.functional as F

def _discretize_bins(x: torch.Tensor, bins: int):
    ranks = torch.argsort(torch.argsort(x, dim=0), dim=0).float()
    q = ranks / (ranks.shape[0] - 1 + 1e-8)
    return torch.clamp((q * bins).long(), 0, bins - 1)

def _nmi_pairwise(a_code: torch.Tensor, b_code: torch.Tensor, bins: int):
    N = a_code.shape[0]
    joint = torch.zeros((bins, bins), device=a_code.device, dtype=torch.float32)
    joint.index_put_((a_code, b_code), torch.ones(N, device=a_code.device), accumulate=True)
    joint = joint / (joint.sum() + 1e-8)
    pa = joint.sum(dim=1, keepdim=True) + 1e-12
    pb = joint.sum(dim=0, keepdim=True) + 1e-12
    mi = (joint * (torch.log(joint + 1e-12) - torch.log(pa) - torch.log(pb))).sum()
    norm = torch.log(torch.tensor(float(bins), device=a_code.device) + 1e-12)
    return (mi / (norm + 1e-12)).clamp(min=0.0)

def normalize_matrix(M: torch.Tensor):
    m = M.clone()
    m = m - m.min()
    m = m / (m.max() + 1e-8)
    return m - 0.5

def aggregate_granules(granules: torch.Tensor, gmask: torch.Tensor):
    k = granules[..., 0]
    b = granules[..., 1]
    s = granules[..., 2]
    T = granules[..., 3].clamp(min=1.0)

    w = T * gmask
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

    k_agg = (w * k).sum(dim=-1)
    b_agg = (w * b).sum(dim=-1)
    s_agg = (w * s).sum(dim=-1)
    return k_agg, b_agg, s_agg

def compute_mic_total_from_granules(granules: torch.Tensor, gmask: torch.Tensor,
                                   bins: int, topk: int, candidate_mult: int):
    k, b, s = aggregate_granules(granules, gmask)
    k0, b0, s0 = k.detach(), b.detach(), s.detach()
    N, D = k0.shape

    # ✅ 修复：val 最后一批可能 N=1，std(unbiased=True) 会 warning 且不稳定
    denom = (k0.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    kk = (k0 - k0.mean(dim=0, keepdim=True)) / denom
    corr = ((kk.t() @ kk) / (max(1, N - 1) + 1e-8)).abs()
    candK = min(D, max(1, topk * candidate_mult))

    k_code = _discretize_bins(k0, bins)
    b_code = _discretize_bins(b0, bins)
    s_code = _discretize_bins(s0, bins)

    mic_trend = torch.zeros((D, D), device=granules.device)
    mic_fluct = torch.zeros((D, D), device=granules.device)

    for u in range(D):
        idx = torch.topk(corr[u], k=candK, largest=True).indices
        au_k = k_code[:, u]
        au_b = b_code[:, u]
        au_s = s_code[:, u]
        for v in idx.tolist():
            if v == u:
                continue
            mi_kk = _nmi_pairwise(au_k, k_code[:, v], bins)
            mi_bb = _nmi_pairwise(au_b, b_code[:, v], bins)
            mi_kb = _nmi_pairwise(au_k, b_code[:, v], bins)
            mi_bk = _nmi_pairwise(au_b, k_code[:, v], bins)
            mic_trend[u, v] = 0.25 * (mi_kk + mi_bb + mi_kb + mi_bk)
            mic_fluct[u, v] = _nmi_pairwise(au_s, s_code[:, v], bins)

    mic_trend = torch.maximum(mic_trend, mic_trend.t())
    mic_fluct = torch.maximum(mic_fluct, mic_fluct.t())
    return mic_trend, mic_fluct

def compute_mic_total_from_mtfig(k: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor,
                                bins: int = 16, topk: int = 16, candidate_mult: int = 4):
    B, D = k.shape
    device = k.device
    T = torch.ones_like(k)
    granules = torch.stack([k, b, sigma, T], dim=-1).unsqueeze(2)
    gmask = torch.ones((B, D, 1), device=device, dtype=torch.float32)
    return compute_mic_total_from_granules(granules, gmask, bins=bins, topk=topk, candidate_mult=candidate_mult)

def build_hypergraph_incidence_from_mic(mic_total: torch.Tensor, topk: int, **kwargs):
    D = mic_total.shape[0]
    E = D
    H = torch.zeros((D, E), device=mic_total.device)
    A = torch.zeros((D, D), device=mic_total.device)

    for u in range(D):
        vals = mic_total[u].clone()
        vals[u] = -1e9
        k = min(int(topk), D - 1)
        if k <= 0:
            H[u, u] = 1.0
            continue
        idx = torch.topk(vals, k=k).indices
        w = F.softmax(mic_total[u, idx], dim=0)

        H[u, u] = 1.0
        H[idx, u] = w
        A[u, idx] = w
        A[idx, u] = w

    return H, A