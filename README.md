# 📈 Window-Scoped Trend Relations for Multivariate Time-Series Analytics

**GHEF — Granule-Hypergraph Enhanced Forecaster**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://huggingface.co/docs/transformers/)
[![Task](https://img.shields.io/badge/Task-Multivariate%20Forecasting-5B8DEF)](#-overview)

This repository contains the **GHEF forecasting implementation** associated with the paper **“Window-Scoped Trend Relations for Multivariate Time-Series Analytics.”** The paper studies how historical time windows define derived trend records, channel affinities, and group memberships, using GHEF as a forecasting consumer.

**🔗 Navigation:** [Contributions](#-main-contributions) · [Method](#-method-overview) · [Datasets](#-datasets) · [Structure](#-repository-structure) · [Requirements](#-requirements) · [Usage](#-quick-start) · [Reproducibility](#-reproducibility-notes)

## 🌟 Main Contributions

- **🔎 Window-scoped query contract.** Characterizes the inputs and identities of derived `Segment`, `Affinity`, and `Member` relations, including source revision, time interval, selected channels, normalization, and construction settings.
- **🕸️ Explicit group semantics.** Separates trend-based pair scoring, mutual-neighbor selection, and group enumeration. The paper shows why the same pair support need not imply the same hypergraph incidence relation and analyzes the work associated with relation construction.
- **🧪 Separated empirical evidence.** Connects the archived GHEF forecasting results with independent finite-graph checks and SQLite experiments on synthetic audit records, without treating those checks as end-to-end GHEF query acceleration.

The reported forecasting study covers **seven benchmark datasets** and **four prediction horizons**. Relative to CVC, the archived results show lower MAE in **22 of 28** dataset–horizon comparisons, with one tie and five losses. These are **reported paper results**, not newly reproduced measurements from this repository.

## 🧩 Method Overview

The GHEF forecasting pipeline has three components:

```text
📥 Historical Multivariate Time Series
                  │
                  ▼
🔹 GSA — Granulation and Semantic Adaptation
   Local trend fitting → MT-FIG descriptors → GPT-2 adaptation
                  │
                  ▼
🔹 HDL — Hypergraph Dependency Learning
   Trend-based association scores → Neighbor selection
   → Incidence-based relational features
                  │
                  ▼
🔹 HGF — Hypergraph-Guided Fusion
   Temporal features + Hypergraph features
   → Cross-attention fusion → Future trend/point prediction
                  │
                  ▼
📊 Multi-Horizon Forecasts (MAE / MSE)
```

- **📉 GSA:** Represents local trends with multi-trend fuzzy information granules (MT-FIGs), including slope, intercept, fluctuation, and duration.
- **🔗 HDL:** Builds channel-dependency representations from trend-based affinities and propagates features through a hypergraph.
- **🧠 HGF:** Fuses temporal and relational representations for multivariate forecasting, using an offline GPT-2 backbone and trainable adaptation components.

> **Scope:** The paper specifies logical trend/affinity/membership relations and discusses group-enumeration policies. The supplied forecasting code uses an approximate, binned association computation and a weighted per-channel incidence construction; it is **not** a native database engine or a complete implementation of the paper's independent CPU/SQLite audits.

## 📚 Datasets

Seven public multivariate time-series benchmarks are used in the paper. The forecasting code loads a CSV file for each dataset from `datasets/<dataset>/<dataset>.csv`.

| 📁 Dataset | 🏷️ Domain | ⏱️ Sampling interval | 🔢 Variables |
|:--|:--|:--|--:|
| `ETTh1` | Transformer temperature | 1 hour | 7 |
| `ETTh2` | Transformer temperature | 1 hour | 7 |
| `ETTm1` | Transformer temperature | 15 minutes | 7 |
| `ETTm2` | Transformer temperature | 15 minutes | 7 |
| `weather` | Meteorological observations | 10 minutes | 21 |
| `electricity` | Electricity consumption | 1 hour | 321 |
| `traffic` | Road traffic occupancy | 1 hour | 862 |

**🎯 Forecast settings in the paper:** lookback length **96**, with horizons **96 / 192 / 336 / 720**. The reported error metrics are **MAE** and **MSE** (lower is better).

**📌 Note:** Your `datasets/` directory also contains `M4/`. It is retained in the folder layout below, but **M4 is not one of the seven datasets in the paper's main benchmark**, and the provided training entry point does not include it by default.

## 🗂️ Repository Structure

The GitHub repository uses three main folders: **`code/`** for the Python implementation, **`datasets/`** for time-series data, and **`gpt2_offline/`** for the local language-model checkpoint. Icons identify the roles of individual files and dataset types.

```text
GHEF/
├── 📘 README.md                                # Project introduction, contributions, and usage
├── 📦 requirements.txt                         # Python dependency list
├── 🔗 .gitattributes                           # Git LFS tracking rules
├── 🚫 .gitignore                               # Ignored caches, outputs, and local files
│
├── 💻 code/                                    # GHEF source code (from GHEF.zip)
│   ├── 🚀 main.py                              # Main training / evaluation entry point
│   ├── ⚙️ config.py                            # Model parameters and experiment settings
│   ├── 📥 data_utils.py                        # Dataset loading, splitting, normalization
│   ├── 🧠 model_ghef.py                        # GHEF model and local GPT-2 backbone
│   ├── 📈 mtfig.py                             # MT-FIG construction and embedding
│   ├── 🕸️ mic_hypergraph.py                    # Trend affinities and hypergraph incidence
│   ├── 🔗 hypergraph_layers.py                 # Hypergraph neural propagation layers
│   ├── 🎯 losses.py                            # Forecasting and auxiliary objectives
│   ├── 🏋️ trainer.py                           # Multi-stage training and evaluation
│   └── 🧱 precompute_granules.py               # Optional MT-FIG precomputation utility
│
├── 📊 datasets/                                # Benchmark data (copied in separately)
│   ├── 🌡️ ETTh1/                               # Hourly transformer temperature
│   ├── 🌡️ ETTh2/                               # Hourly transformer temperature
│   ├── ⏱️ ETTm1/                               # 15-min transformer temperature
│   ├── ⏱️ ETTm2/                               # 15-min transformer temperature
│   ├── 🌦️ weather/                             # Meteorological observations
│   ├── ⚡ electricity/                         # Electricity consumption
│   ├── 🚦 traffic/                             # Road traffic occupancy
│   └── 📈 M4/                                  # Extra dataset; not used in the main 7 benchmarks
│
├── 🤗 gpt2_offline/                            # Offline GPT-2 files (copied in separately)
│   ├── ⚙️ config.json                          # Model architecture configuration
│   ├── 🧠 model.safetensors                    # Pretrained model weights (Git LFS)
│   ├── 🧩 merges.txt                           # Byte-pair encoding merge rules
│   ├── 🔤 tokenizer.json                       # Tokenizer definition
│   ├── 🔧 tokenizer_config.json                # Tokenizer settings
│   └── 🔠 vocab.json                           # Vocabulary lookup
│
└── 📂 outputs/                                 # Created during training (not committed)
```

> 📌 **Directory note:** The `code/` scripts, `README.md`, dependency file, and Git configuration are included in the starter archive. Copy your actual `datasets/` and `gpt2_offline/` contents into the indicated paths. The `outputs/` directory is generated when experiments run; it does not need to be uploaded.

> 🗃️ **Dataset file convention:** Each supported forecasting dataset is expected at `datasets/<dataset>/<dataset>.csv`, such as `datasets/ETTh1/ETTh1.csv`. Dataset subfolder contents are not enumerated above because only their top-level folder names were provided.

## ⚙️ Requirements

**🖥️ Recommended environment**

- 🐍 **Python 3.10+**
- 🔥 **PyTorch 2.x** (install a build compatible with your CUDA driver for GPU training)
- 🤗 **Hugging Face Transformers 4.x**
- 🧮 **NumPy**, **Pandas**, and **tqdm**
- 🚀 An NVIDIA GPU is recommended for the full forecasting experiments, particularly high-dimensional datasets.

Install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> For CUDA-specific PyTorch wheels, follow the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining dependencies. `requirements.txt` specifies practical version ranges; it is not an exact lockfile for the environment used to obtain the paper's archived results.

## 🚀 Quick Start

### 1. 📥 Prepare the data

Ensure each benchmark has the expected CSV path, for example:

```text
🌡️ datasets/ETTh1/ETTh1.csv
⏱️ datasets/ETTm2/ETTm2.csv
⚡ datasets/electricity/electricity.csv
```

The loader reads numeric channels from each CSV and drops a nonnumeric first column (such as a timestamp) when present.

### 2. 🤗 Prepare the offline GPT-2 checkpoint

Ensure that `gpt2_offline/config.json` and `gpt2_offline/model.safetensors` are available. The model is loaded from the local directory using `local_files_only=True`; training does **not** download GPT-2 automatically.

### 3. ▶️ Run a single experiment

Run this command **from the repository root**:

```bash
python code/main.py \
  --data_root ./datasets \
  --gpt2_name ./gpt2_offline \
  --out_dir ./outputs \
  --datasets ETTh1 \
  --horizons 96 \
  --lookback 96 \
  --device cuda
```

For a CPU-only installation, change `--device cuda` to `--device cpu` (expect much longer execution time) and, if needed, add `--num_workers 0 --no_amp`.

### 4. 🧪 Run the seven-dataset benchmark

```bash
python code/main.py \
  --data_root ./datasets \
  --gpt2_name ./gpt2_offline \
  --out_dir ./outputs \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather electricity traffic \
  --horizons 96 192 336 720 \
  --lookback 96 \
  --epochs 100 \
  --device cuda
```

This executes the available training pipeline for the selected dataset–horizon combinations. **It is not guaranteed to reproduce the archived paper numbers**; see the configuration differences below. High-dimensional datasets may require substantial GPU memory and training time.

### 5. 🔧 Useful command-line options

| ⚙️ Argument | 📝 Meaning | 💡 Code default |
|:--|:--|:--|
| `--datasets` | One or more benchmark names | All seven benchmarks |
| `--horizons` | Prediction lengths | `96 192 336 720` |
| `--lookback` | Input sequence length | `512` |
| `--batch` | Batch size | `32` |
| `--lr` | Learning rate | `2e-4` |
| `--epochs` | Stage-3 maximum epochs | `150` |
| `--stage1_epochs` | Stage-1 epochs | `20` |
| `--stage2_epochs` | Stage-2 epochs | `20` |
| `--patience` | Early-stopping patience | `10` |
| `--topk` | Channel-association neighbor budget | `16` |
| `--device` | Training device | `cuda` |
| `--no_stage1` / `--no_stage2` / `--no_stage3` | Skip the named stage | All stages enabled |

**💡 Tip:** Detailed settings, including the GPT-2 layer count, model adaptation, loss weights, and dataset-specific overrides, are defined in [`code/config.py`](code/config.py). Skipping a training stage is **not** the same as loading an existing checkpoint for test-only evaluation.

### 6. 🧱 Optional MT-FIG precomputation

```bash
python code/precompute_granules.py \
  --data_root ./datasets \
  --dataset ETTh1 \
  --out_dir ./granule_cache \
  --lookback 96 \
  --device cuda
```

This exports `.pt` files containing granules and masks for each split. It is a **separate utility**; `code/main.py` does not read these exports as a prerequisite for training.

## 📁 Outputs and Evaluation

After running the main script, prediction errors and checkpoints are written under the selected `--out_dir` (default `./outputs`):

```text
📂 outputs/
├── 🌡️ ETTh1/
│   └── ⏳ H96/
│       ├── 💾 stage1_best.pt
│       ├── 💾 stage2_best.pt
│       └── 💾 stage3_best.pt
├── 📊 results_ETTh1.json
└── 📋 results_all.json
```

- 📉 **MAE** — Mean Absolute Error.
- 📊 **MSE** — Mean Squared Error.
- 💾 `results_<dataset>.json` stores per-horizon scores and their arithmetic average.
- 🗃️ `results_all.json` consolidates the evaluated datasets.
- 🧠 `stage*_best.pt` checkpoints store the selected validation-best model weights for each enabled training stage.

## 🔬 Reproducibility Notes

The paper and the present code snapshot are **not configuration-identical**. Before attempting a matched replication, check at least these differences:

1. **📏 Lookback:** The paper reports `96`; the code defaults to `512` and overrides the lookback to `192` for `electricity` and `96` for `traffic`.
2. **🧠 GPT-2 layers:** The paper describes the first **6** layers; `code/config.py` currently sets `gpt2_use_first_n_layers = 8`. This setting is not a CLI argument.
3. **🕒 Training budget:** The paper reports up to **100** epochs; the current Stage-3 code default is **150**, configurable via `--epochs`.
4. **📐 Non-ETT splitting:** The code uses a **70% / 10% / 20%** train/validation/test split for non-ETT datasets, whereas the paper's reported non-ETT split sizes correspond to approximately **70% / 20% / 10%**.
5. **🕸️ Group construction:** The current code computes a discretized, candidate-pruned association approximation and weighted incidence; it does not implement the full logical clique-enumeration policy analyzed in the paper.
6. **🗄️ Audit scope:** The independent finite-graph and SQLite audit programs referenced in the paper are **not included** in the supplied 10-file code archive.

Accordingly, the included commands document **how to run this released code**, while the archived tables in the paper should be read as separately reported evidence.

## ☁️ Uploading Large Files to GitHub

The offline `model.safetensors` checkpoint shown in the provided directory is approximately **475 MiB**, above GitHub's ordinary **100 MiB per-file** Git limit. Use [Git Large File Storage (Git LFS)](https://git-lfs.com/) **before** adding the model checkpoint:

```bash
git lfs install
git lfs track "gpt2_offline/*.safetensors"
git add .gitattributes
```

Use Git LFS for oversized dataset files as needed. Collaborators who clone the repository should also have Git LFS installed and run `git lfs pull` if large files were not downloaded automatically.

## 📝 Citation

If this repository is useful for your research, please cite the corresponding manuscript:

```bibtex
@article{wang2027windowscoped,
  title   = {Window-Scoped Trend Relations for Multivariate Time-Series Analytics},
  author  = {Wang, Anna and Zhang, Chao and Li, Wentao and Ding, Weiping and Pedersen, Torben Bach and Aggarwal, Charu C. and Pedrycz, Witold},
  journal = {Proceedings of the VLDB Endowment},
  year    = {2027}
}
```

---

**📬 Research use:** This repository provides the GHEF forecasting code, the expected benchmark data layout, and instructions for local GPT-2 loading. Refer to the manuscript for the logical query contract, structural-audit methodology, and archived evaluation results.
