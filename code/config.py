from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class ExpConfig:
    # =======================
    # Data / Loader
    # =======================
    data_root: str = "/data/LiFeiFei/GHEF/datasets/"
    datasets: Optional[List[str]] = None
    split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2)  # train/val/test
    lookback: int = 512
    horizons: Optional[List[int]] = None

    # CSV parsing / scaling
    zscore_per_channel: bool = True
    drop_first_column_if_non_numeric: bool = True

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True

    # =======================
    # Device / Seed
    # =======================
    seed: int = 42
    device: str = "cuda"

    # =======================
    # Model (offline GPT2)
    # =======================
    gpt2_name: str = "D:\python\projects\TFS\gpt2_offline"
    gpt2_use_first_n_layers: int = 8
    d_model: int = 768

    # ===== MT-FIG =====
    n_granules: int = 4
    mtfig_embed_dim: int = 768

    # ===== MIC-driven hypergraph =====
    topk_hyper: int = 16
    mic_bins: int = 16
    mic_candidate_mult: int = 4
    use_geo_hypergraph: bool = False
    hyperedge_per_node: bool = True

    # ===== mask enhanced attn（保留字段，模型里可不用）=====
    use_mask_enhanced_attn: bool = True
    mask_causal: bool = True
    mask_periodic: bool = True
    mask_trend: bool = True

    # ===== LoRA =====
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    # =======================
    # Prompt + RevIN + time features
    # =======================
    use_metadata_prompt: bool = True
    prompt_position: str = "prepend"   # "prepend" or "append"
    max_prompt_tokens: int = 48
    dataset_freq_map: Dict[str, str] = None
    use_revin: bool = True
    use_time_features: bool = True

    # =======================
    # Training
    # =======================
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 5e-4
    max_epochs: int = 150
    early_stop_patience: int = 10
    grad_clip: float = 1.0
    use_amp: bool = True

    # Scheduler
    use_scheduler: bool = True
    warmup_ratio: float = 0.10
    min_lr_ratio: float = 0.05

    # Loss weights (stage3)
    alpha_mtfig: float = 0.15
    beta_hyper: float = 1e-5
    gamma_consistency: float = 0.0
    dtw_band: int = 16

    # stage3 预测 loss 设置（losses.py 会读 aux 里的这俩）
    pred_loss: str = "mix"
    smoothl1_beta: float = 0.2
    mix_lambda: float = 0.85

    # =======================
    # Horizon-stable training
    # =======================
    use_time_weight: bool = True
    time_weight_lambda: float = 1.0
    time_weight_power: float = 1.7
    ms_scales: Tuple[int, ...] = (2, 4, 8)
    ms_weight: float = 0.08

    # 分组学习率
    lr_gpt: float = 1e-5

    # decoder
    dec_heads: int = 8
    dec_layers: int = 2

    # 其它稳定项
    hyper_scale_init: float = 0.01
    stage1_recon_len: int = 96

    # =======================
    # Stages
    # =======================
    run_stage1: bool = True
    run_stage2: bool = True
    run_stage3: bool = True
    stage1_epochs: int = 20
    stage2_epochs: int = 20

    # =======================
    # Logging / Outputs
    # =======================
    out_dir: str = "./outputs"

    # =======================
    # Dataset-specific overrides
    # 只对 electricity / traffic 生效，其它数据集完全不改
    # =======================
    dataset_special_overrides: Dict[str, Dict] = None

    def __post_init__(self):
        if self.dataset_freq_map is None:
            self.dataset_freq_map = {
                "ETTh1": "1 hour",
                "ETTh2": "1 hour",
                "ETTm1": "15 minutes",
                "ETTm2": "15 minutes",
                "weather": "10 minutes",
                "electricity": "1 hour",
                "traffic": "1 hour",
            }

        if self.dataset_special_overrides is None:
            self.dataset_special_overrides = {
                # 这两个高维数据集会导致 GPT2 token 长度爆掉：
                # total_tokens ≈ lookback + D*n_granules + prompt_tokens
                # 因此只对它们单独缩短 lookback / granules / prompt，并设置 token budget。
                "electricity": {
                    "lookback": 192,
                    "n_granules": 1,
                    "use_metadata_prompt": False,
                    "max_prompt_tokens": 0,
                    "gpt2_token_budget": 1024,
                    "mtfig_token_budget": 768,
                },
                "traffic": {
                    "lookback": 96,
                    "n_granules": 1,
                    "use_metadata_prompt": False,
                    "max_prompt_tokens": 0,
                    "gpt2_token_budget": 1024,
                    "mtfig_token_budget": 896,
                },
            }
