# model_ghef.py
import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from transformers import GPT2Model

from mtfig import fit_linear_trend_params, build_mtfig_granules, MTFIGGranuleEmbed
from mic_hypergraph import compute_mic_total_from_mtfig, build_hypergraph_incidence_from_mic
from hypergraph_layers import HypergraphConv


class RevIN(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, d))
        self.beta = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x: torch.Tensor, mode: str, stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if mode == "norm":
            mu = x.mean(dim=1, keepdim=True)
            var = (x - mu).pow(2).mean(dim=1, keepdim=True)
            sigma = torch.sqrt(var + self.eps)
            x_hat = (x - mu) / sigma
            x_hat = x_hat * self.gamma + self.beta
            return x_hat, (mu, sigma)
        elif mode == "denorm":
            assert stats is not None
            mu, sigma = stats
            x = (x - self.beta) / (self.gamma + 1e-8)
            x = x * sigma + mu
            return x
        else:
            raise ValueError("mode must be 'norm' or 'denorm'.")


def sinusoidal_time_features(L: int, device: torch.device, dtype: torch.dtype):
    # 3 periodicities -> 6 dims
    pos = torch.arange(L, device=device, dtype=dtype).view(1, L, 1)
    periods = torch.tensor([24.0, 48.0, 96.0], device=device, dtype=dtype).view(1, 1, 3)
    ang = 2.0 * math.pi * pos / (periods + 1e-6)
    feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return feats


class CrossAttnDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(query=q, key=kv, value=kv, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(q + attn_out)
        x2 = self.ffn(x)
        x = self.ln2(x + x2)
        return x


class BytePromptEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 48):
        super().__init__()
        self.max_len = int(max_len)
        self.emb = nn.Embedding(256, d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def encode(self, text: str, device: torch.device) -> torch.Tensor:
        b = text.encode("utf-8", errors="ignore")[: self.max_len]
        if len(b) == 0:
            return torch.empty(1, 0, self.emb.embedding_dim, device=device)
        ids = torch.tensor(list(b), dtype=torch.long, device=device).view(1, -1)
        x = self.emb(ids)
        return self.proj(x)


# =========================
# LoRA for GPT2 Conv1D
# =========================
class LoRAConv1D(nn.Module):
    """
    Wrap transformers.pytorch_utils.Conv1D (used in GPT2) with low-rank update:
      y = base(x) + scale * (x @ A @ B)
    where A: [in, r], B: [r, out]
    """
    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scale = float(alpha) / float(max(1, r))
        self.drop = nn.Dropout(float(dropout))
        # Conv1D in GPT2 exposes .weight [in, out], .bias [out]
        in_dim = base.weight.shape[0]
        out_dim = base.weight.shape[1]
        self.A = nn.Parameter(torch.zeros(in_dim, self.r))
        self.B = nn.Parameter(torch.zeros(self.r, out_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        x2 = self.drop(x)
        # x: [B,T,in] -> [B,T,out]
        lora = (x2 @ self.A) @ self.B
        return y + self.scale * lora


def apply_lora_and_freeze_gpt2(gpt2: GPT2Model, freeze_until: int, unfreeze_from: int,
                              use_lora: bool, r: int, alpha: float, dropout: float):
    """
    Paper-style hierarchical protection:
      - layers < freeze_until: fully frozen
      - freeze_until <= layers < unfreeze_from: frozen base weights, train LoRA adapters
      - layers >= unfreeze_from: fully trainable
    """
    freeze_until = int(freeze_until)
    unfreeze_from = int(unfreeze_from)

    for li, block in enumerate(gpt2.h):
        if li < freeze_until:
            for p in block.parameters():
                p.requires_grad = False
            continue

        if li >= unfreeze_from:
            for p in block.parameters():
                p.requires_grad = True
            continue

        # middle: freeze base, add LoRA on key modules
        for p in block.parameters():
            p.requires_grad = False

        if use_lora:
            # attn projections
            if hasattr(block.attn, "c_attn") and not isinstance(block.attn.c_attn, LoRAConv1D):
                block.attn.c_attn = LoRAConv1D(block.attn.c_attn, r=r, alpha=alpha, dropout=dropout)
            if hasattr(block.attn, "c_proj") and not isinstance(block.attn.c_proj, LoRAConv1D):
                block.attn.c_proj = LoRAConv1D(block.attn.c_proj, r=r, alpha=alpha, dropout=dropout)
            # MLP projections
            if hasattr(block.mlp, "c_fc") and not isinstance(block.mlp.c_fc, LoRAConv1D):
                block.mlp.c_fc = LoRAConv1D(block.mlp.c_fc, r=r, alpha=alpha, dropout=dropout)
            if hasattr(block.mlp, "c_proj") and not isinstance(block.mlp.c_proj, LoRAConv1D):
                block.mlp.c_proj = LoRAConv1D(block.mlp.c_proj, r=r, alpha=alpha, dropout=dropout)

            # ensure LoRA params trainable
            for m in block.modules():
                if isinstance(m, LoRAConv1D):
                    for p in m.parameters():
                        p.requires_grad = True


class BiMaskedCrossFusion(nn.Module):
    """
    Phase II: Bidirectional cross-attn with A_hyper (t->h) and M_temporal (h->t).
    We implement it between:
      - H_t: MT-FIG token representations (length = D*G)
      - H_h: hypergraph node embeddings (length = D)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn_t2h = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_h2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_t = nn.LayerNorm(d_model)
        self.ln_h = nn.LayerNorm(d_model)

    @staticmethod
    def _build_t2h_mask(A_hyper: torch.Tensor, D: int, G: int, device: torch.device):
        # A_hyper: [D,D] (float), allow if >0 or self
        allow = (A_hyper > 0)
        allow.fill_diagonal_(True)
        # mask shape [D*G, D]
        allow_tokens = allow.repeat_interleave(G, dim=0)  # [D*G, D]
        # attn_mask: float with -inf for disallowed
        mask = torch.zeros((D * G, D), device=device)
        mask[~allow_tokens] = float("-inf")
        return mask

    @staticmethod
    def _build_h2t_mask(D: int, G: int, device: torch.device, band: int = 1):
        """
        M_temporal approximation: each hyper-node (channel u) attends only to its own granules
        (and optionally +/- band granules, but since tokens are ordered by granule within channel,
        we just allow all tokens of the same channel).
        """
        mask = torch.zeros((D, D * G), device=device)
        # disallow tokens not belonging to the same channel
        for u in range(D):
            allowed = torch.zeros(D * G, device=device, dtype=torch.bool)
            start = u * G
            allowed[start:start + G] = True
            mask[u, ~allowed] = float("-inf")
        return mask

    def forward(self, H_t: torch.Tensor, H_h: torch.Tensor, A_hyper: torch.Tensor, D: int, G: int):
        # H_t: [B, D*G, d], H_h: [B, D, d]
        device = H_t.device
        t2h_mask = self._build_t2h_mask(A_hyper.detach(), D=D, G=G, device=device)  # [DG, D]
        h2t_mask = self._build_h2t_mask(D=D, G=G, device=device)                    # [D, DG]

        # t->h
        out_t2h, _ = self.attn_t2h(query=H_t, key=H_h, value=H_h, attn_mask=t2h_mask, need_weights=False)
        Ht = self.ln_t(H_t + out_t2h)

        # h->t
        out_h2t, _ = self.attn_h2t(query=H_h, key=Ht, value=Ht, attn_mask=h2t_mask, need_weights=False)
        Hh = self.ln_h(H_h + out_h2t)
        return Ht, Hh


class GHEF(nn.Module):
    """
    Paper-aligned upgrades (priority path):
      1) Phase II: bidirectional cross-attn fusion with A_hyper and M_temporal (approx) between MT-FIG tokens and hyper nodes.
      2) Phase III: two-stage prediction (predict future trend params -> reconstruct) + residual refinement.
      3) LoRA + hierarchical freezing for GPT2 blocks (granular mask is approximated by extra temporal features; GPT2 does not accept bias mask).
    """

    def __init__(self, D: int, cfg):
        super().__init__()
        self.D = int(D)
        self.cfg = cfg
        self.d_model = int(getattr(cfg, "d_model", 768))

        # ---- GPT2 backbone ----
        self.gpt2 = GPT2Model.from_pretrained(cfg.gpt2_name, local_files_only=True)
        self.gpt2.config.use_cache = False
        n_use = int(getattr(cfg, "gpt2_use_first_n_layers", 8))
        n_use = max(1, min(n_use, len(self.gpt2.h)))
        self.gpt2.h = nn.ModuleList(list(self.gpt2.h)[:n_use])

        # ---- LoRA + freeze policy ----
        apply_lora_and_freeze_gpt2(
            self.gpt2,
            freeze_until=int(getattr(cfg, "freeze_until", 0)),
            unfreeze_from=int(getattr(cfg, "unfreeze_from", n_use)),
            use_lora=bool(getattr(cfg, "use_lora", False)),
            r=int(getattr(cfg, "lora_r", 8)),
            alpha=float(getattr(cfg, "lora_alpha", 16.0)),
            dropout=float(getattr(cfg, "lora_dropout", 0.05)),
        )

        # ---- prompt ----
        self.use_metadata_prompt = bool(getattr(cfg, "use_metadata_prompt", False))
        self.prompt_position = str(getattr(cfg, "prompt_position", "prepend")).lower()
        self.max_prompt_tokens = int(getattr(cfg, "max_prompt_tokens", 48))
        self.prompt_encoder = BytePromptEncoder(self.d_model, max_len=self.max_prompt_tokens)

        # ---- numeric embedding (time tokens) ----
        self.use_time_features = bool(getattr(cfg, "use_time_features", True))
        in_dim = self.D + (6 if self.use_time_features else 0)
        drop = float(getattr(cfg, "dropout", 0.05))

        self.x_proj = nn.Sequential(
            nn.Linear(in_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(self.d_model, self.d_model),
        )

        # ---- MT-FIG tokens ----
        self.mtfig_gran_embed = MTFIGGranuleEmbed(self.d_model)

        # ---- Hypergraph ----
        self.hyper_conv = HypergraphConv(self.d_model, self.d_model)

        # ---- Phase II cross fusion ----
        self.fusion = BiMaskedCrossFusion(self.d_model, n_heads=int(getattr(cfg, "dec_heads", 8)), dropout=drop)

        # ---- RevIN ----
        self.use_revin = bool(getattr(cfg, "use_revin", True))
        self.revin = RevIN(self.D) if self.use_revin else None

        # ---- Decoder for refinement ----
        n_heads = int(getattr(cfg, "dec_heads", 8))
        n_layers = max(1, int(getattr(cfg, "dec_layers", 2)))
        self.horizon_pos = nn.Parameter(torch.randn(1, 1024, self.d_model) * 0.01)
        self.decoder_blocks = nn.ModuleList(
            [CrossAttnDecoderBlock(self.d_model, n_heads, dropout=drop) for _ in range(n_layers)]
        )

        # refinement head (delta)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(self.d_model, self.D),
        )

        # Phase III param head: predict future (k,b,sigma) per channel from pooled decoder states
        self.param_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(self.d_model, self.D * 3),
        )

        # gate blending of reconstructed vs base+delta
        self.recon_gate = nn.Parameter(torch.tensor(0.2))

        # hyper injection strength (small)
        init_scale = float(getattr(cfg, "hyper_scale_init", 0.01))
        self.hyper_scale = nn.Parameter(torch.tensor(init_scale))

        self.dataset_name = None
        self._time_feat_cache = {}
        self._prompt_cache = {}

    def set_dataset_name(self, name: str):
        self.dataset_name = name
        self._prompt_cache.clear()

    def _get_dataset_runtime_settings(self) -> Dict:
        ds = (self.dataset_name or "").lower()
        out = {
            "gpt2_token_budget": int(getattr(self.cfg, "gpt2_token_budget", getattr(self.gpt2.config, "n_positions", 1024))),
            "mtfig_token_budget": int(getattr(self.cfg, "mtfig_token_budget", 10**9)),
            "force_disable_prompt": False,
        }
        if ds in ("electricity", "traffic"):
            out["force_disable_prompt"] = not bool(getattr(self.cfg, "use_metadata_prompt", False)) or int(getattr(self.cfg, "max_prompt_tokens", 0)) <= 0
        return out

    def _compress_mt_tokens(self, mt_tokens: torch.Tensor, gmask: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Only used for electricity / traffic when D is very large.
        Compress [B, D*G, d] -> [B, <=target_len, d] by averaging tokens inside each channel group.
        This keeps other datasets untouched.
        """
        B, DG, C = mt_tokens.shape
        G = int(gmask.shape[-1])
        D = int(gmask.shape[1])
        if target_len <= 0 or DG <= target_len or G <= 0:
            return mt_tokens

        mt_tokens = mt_tokens.view(B, D, G, C)
        weights = gmask.to(dtype=mt_tokens.dtype, device=mt_tokens.device).unsqueeze(-1)
        denom = weights.sum(dim=2, keepdim=False).clamp_min(1e-6)
        ch_tokens = (mt_tokens * weights).sum(dim=2) / denom

        if ch_tokens.shape[1] <= target_len:
            return ch_tokens

        # uniform downsample on channel dimension
        idx = torch.linspace(0, ch_tokens.shape[1] - 1, steps=target_len, device=ch_tokens.device)
        idx = idx.round().long().clamp(0, ch_tokens.shape[1] - 1)
        return ch_tokens[:, idx, :]

    def _fit_token_budget(self, mt_tokens: torch.Tensor, num_emb: torch.Tensor, prompt_emb: Optional[torch.Tensor], runtime_cfg: Dict):
        max_pos = int(getattr(self.gpt2.config, "n_positions", 1024))
        token_budget = int(runtime_cfg.get("gpt2_token_budget", max_pos))
        token_budget = max(16, min(token_budget, max_pos))

        if prompt_emb is not None and bool(runtime_cfg.get("force_disable_prompt", False)):
            prompt_emb = None

        mt_budget = int(runtime_cfg.get("mtfig_token_budget", mt_tokens.shape[1]))
        if mt_budget > 0 and mt_tokens.shape[1] > mt_budget:
            mt_tokens = mt_tokens[:, :mt_budget, :]

        prompt_len = 0 if prompt_emb is None else int(prompt_emb.shape[1])
        total = int(mt_tokens.shape[1] + num_emb.shape[1] + prompt_len)
        if total <= token_budget:
            return mt_tokens, num_emb, prompt_emb

        # preserve full numeric tokens first; trim prompt, then MT-FIG, and only finally numeric tokens.
        overflow = total - token_budget
        if prompt_emb is not None and overflow > 0:
            keep = max(0, prompt_emb.shape[1] - overflow)
            prompt_emb = prompt_emb[:, :keep, :] if keep > 0 else None
            prompt_len = 0 if prompt_emb is None else int(prompt_emb.shape[1])
            total = int(mt_tokens.shape[1] + num_emb.shape[1] + prompt_len)
            overflow = max(0, total - token_budget)

        if overflow > 0:
            keep_mt = max(0, mt_tokens.shape[1] - overflow)
            mt_tokens = mt_tokens[:, :keep_mt, :] if keep_mt > 0 else mt_tokens[:, :0, :]
            total = int(mt_tokens.shape[1] + num_emb.shape[1] + (0 if prompt_emb is None else prompt_emb.shape[1]))
            overflow = max(0, total - token_budget)

        if overflow > 0:
            keep_num = max(16, num_emb.shape[1] - overflow)
            num_emb = num_emb[:, -keep_num:, :]

        return mt_tokens, num_emb, prompt_emb

    def _build_prompt_text(self, lookback: int, horizon: int) -> str:
        ds = self.dataset_name or "unknown"
        freq_map = getattr(self.cfg, "dataset_freq_map", {}) or {}
        freq = freq_map.get(ds, "unknown")
        return (
            f"Dataset: {ds}. Sampling: {freq}. "
            f"Lookback: {lookback}. Horizon: {horizon}. "
            f"Task: multivariate time series forecasting."
        )

    def _get_time_features_cached(self, L: int, device: torch.device, dtype: torch.dtype):
        key = (int(L), str(device), str(dtype))
        out = self._time_feat_cache.get(key, None)
        if out is None:
            out = sinusoidal_time_features(L, device=device, dtype=dtype)
            self._time_feat_cache[key] = out
        return out

    def _get_prompt_cached(self, lookback: int, horizon: int, device: torch.device):
        text = self._build_prompt_text(lookback=lookback, horizon=horizon)

        # IMPORTANT:
        # During training, prompt_encoder outputs are connected to the current autograd graph
        # (Embedding/Linear parameters are trainable). Reusing a cached prompt embedding tensor
        # across iterations will reuse the old graph and trigger:
        # "Trying to backward through the graph a second time".
        # Therefore, only cache prompt embeddings when we are not training (or when the prompt
        # encoder is effectively frozen). During training we always recompute them.
        trainable_prompt = any(p.requires_grad for p in self.prompt_encoder.parameters())
        if self.training and trainable_prompt:
            return self.prompt_encoder.encode(text, device=device)

        key = (text, str(device))
        out = self._prompt_cache.get(key, None)
        if out is None:
            out = self.prompt_encoder.encode(text, device=device)
            self._prompt_cache[key] = out
        return out

    def forward(self, x: torch.Tensor, horizon: int):
        """
        x: [B, L, D]
        return: y_hat [B, H, D], aux dict
        """
        B, L, D = x.shape
        device = x.device
        aux: Dict = {}

        # RevIN
        if self.use_revin:
            x_in, stats = self.revin(x, mode="norm")
        else:
            x_in, stats = x, None

        # time features
        if self.use_time_features:
            tf = sinusoidal_time_features(L, device=device, dtype=x_in.dtype).repeat(B, 1, 1)
            x_feat = torch.cat([x_in, tf], dim=-1)
        else:
            x_feat = x_in

        num_emb = self.x_proj(x_feat)  # [B,L,d]

        # trend params from input (stage2 alignment)
        k_hat, b_hat, s_hat, _T = fit_linear_trend_params(x_in)
        aux["k_hat"], aux["b_hat"], aux["s_hat"] = k_hat, b_hat, s_hat

        # MT-FIG granules -> tokens (length DG)
        granules, gmask = build_mtfig_granules(x_in, self.cfg)  # [B,D,G,4], [B,D,G]
        G = granules.shape[2]
        mt_tokens = self.mtfig_gran_embed(granules)            # [B, D*G, d]

        runtime_cfg = self._get_dataset_runtime_settings()
        if (self.dataset_name or "").lower() in ("electricity", "traffic"):
            mt_budget = int(runtime_cfg.get("mtfig_token_budget", mt_tokens.shape[1]))
            mt_tokens = self._compress_mt_tokens(mt_tokens, gmask, target_len=mt_budget)

        # MIC hypergraph (channel graph)
        with torch.no_grad():
            mic_trend, mic_fluct = compute_mic_total_from_mtfig(
                k_hat, b_hat, s_hat,
                bins=int(getattr(self.cfg, "mic_bins", 16)),
                topk=int(getattr(self.cfg, "topk_hyper", 16)),
                candidate_mult=int(getattr(self.cfg, "mic_candidate_mult", 4)),
            )
            mic_total = mic_trend + mic_fluct
            H_inc, A_hyper = build_hypergraph_incidence_from_mic(mic_total, topk=int(getattr(self.cfg, "topk_hyper", 16)))
        aux["A_hyper"] = A_hyper

        # hypergraph node embeddings: use per-channel statistics from x_in (fast, informative)
        ch_mean = x_in.mean(dim=1)           # [B,D]
        ch_std = x_in.std(dim=1, unbiased=False)  # [B,D]
        ch_feat = torch.stack([ch_mean, ch_std], dim=-1)  # [B,D,2]
        ch_emb = torch.tanh(ch_feat).repeat(1, 1, self.d_model // 2)  # cheap expand
        if ch_emb.shape[-1] != self.d_model:
            ch_emb = torch.cat([ch_emb, torch.zeros(B, D, self.d_model - ch_emb.shape[-1], device=device, dtype=x_in.dtype)], dim=-1)

        ch_emb2, _ = self.hyper_conv(ch_emb, H_inc)          # [B,D,d]
        # Phase II fusion (most important)
        mt_tokens_fused, ch_emb_fused = self.fusion(mt_tokens, ch_emb2, A_hyper=A_hyper, D=D, G=G)

        # inject fused hyper info into temporal tokens cheaply (broadcast)
        scale = torch.clamp(self.hyper_scale, 0.0, 0.2)
        num_emb = num_emb + scale * ch_emb_fused.mean(dim=1, keepdim=True)

        # prompt (paper uses prompt both sides; we keep compat but allow both if set in cfg)
        prompt_emb = None
        use_prompt_now = self.use_metadata_prompt and (not bool(runtime_cfg.get("force_disable_prompt", False)))
        if use_prompt_now:
            pe = self._get_prompt_cached(lookback=L, horizon=int(horizon), device=device)
            if pe.shape[1] > 0:
                prompt_emb = pe.repeat(B, 1, 1)

        mt_tokens_fused, num_emb, prompt_emb = self._fit_token_budget(
            mt_tokens_fused, num_emb, prompt_emb, runtime_cfg=runtime_cfg
        )

        parts, masks = [], []

        def _mask(tlen: int):
            return torch.ones((B, tlen), device=device, dtype=torch.long)

        # prompt both sides if cfg.prompt_position == "both"
        pos = self.prompt_position
        if pos not in ("prepend", "append", "both"):
            pos = "prepend"

        if pos in ("prepend", "both"):
            if prompt_emb is not None:
                parts.append(prompt_emb); masks.append(_mask(prompt_emb.shape[1]))

        # MT-FIG tokens (fused) then numeric tokens
        parts.append(mt_tokens_fused); masks.append(_mask(mt_tokens_fused.shape[1]))
        parts.append(num_emb);        masks.append(_mask(num_emb.shape[1]))

        if pos in ("append", "both"):
            if prompt_emb is not None:
                parts.append(prompt_emb); masks.append(_mask(prompt_emb.shape[1]))

        inp_emb = torch.cat(parts, dim=1)
        attn_mask = torch.cat(masks, dim=1)

        out = self.gpt2(inputs_embeds=inp_emb, attention_mask=attn_mask)
        h = out.last_hidden_state  # [B, S, d]

        # extract numeric token hidden states (last numeric token block)
        L_eff = int(num_emb.shape[1])
        start_num = h.shape[1] - L_eff
        h_num = h[:, start_num:start_num + L_eff, :]  # [B,L_eff,d]

        # ========= Phase III: two-stage prediction =========
        Hh = int(horizon)
        # base (persistence) in normalized space
        base = x_in[:, -1:, :].repeat(1, Hh, 1)  # [B,H,D]

        # refinement via cross-attn decoder over temporal memory
        q = self.horizon_pos[:, :Hh, :].to(device).repeat(B, 1, 1)
        dec = q
        for blk in self.decoder_blocks:
            dec = blk(dec, h_num)

        delta = self.delta_head(dec)  # [B,H,D]

        # predict future trend params from pooled decoder state
        pooled = dec.mean(dim=1)      # [B,d]
        params = self.param_head(pooled).view(B, self.D, 3)
        k_f = params[:, :, 0]
        b_f = params[:, :, 1]
        s_f = params[:, :, 2].abs() + 1e-6
        aux["k_f_hat"], aux["b_f_hat"], aux["s_f_hat"] = k_f, b_f, s_f

        t = torch.arange(Hh, device=device, dtype=x_in.dtype).view(1, Hh, 1)
        y_recon = k_f.unsqueeze(1) * t + b_f.unsqueeze(1)  # [B,H,D]

        gate = torch.sigmoid(self.recon_gate).clamp(0.0, 1.0)
        y_hat = base + delta + gate * (y_recon - base)

        if self.use_revin:
            y_hat = self.revin(y_hat, mode="denorm", stats=stats)

        # loss control
        aux["pred_loss"] = str(getattr(self.cfg, "pred_loss", "mix"))
        aux["smoothl1_beta"] = float(getattr(self.cfg, "smoothl1_beta", 0.2))
        aux["mix_lambda"] = float(getattr(self.cfg, "mix_lambda", 0.85))

        # L_mtfig will be set in trainer (uses y_true); keep tiny fallback to avoid None
        aux["L_mtfig"] = torch.tensor(0.0, device=device, dtype=x_in.dtype)

        return y_hat, aux
