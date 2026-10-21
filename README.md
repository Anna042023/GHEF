# 📈 Window-Scoped Trend Relations for Multivariate Time-Series Analytics

**GHEF — Granule-Hypergraph Enhanced Forecaster**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://huggingface.co/docs/transformers/)
[![Task](https://img.shields.io/badge/Task-Multivariate%20Forecasting-5B8DEF)](#-overview)

This repository contains the **GHEF forecasting implementation** associated with the paper **“Window-Scoped Trend Relations for Multivariate Time-Series Analytics.”** The paper studies how historical time windows define derived trend records, channel affinities, and group memberships, using GHEF as a forecasting consumer.

**🔗 Navigation:** [Contributions](#-main-contributions) · [Method](#-method-overview) · [Datasets](#-datasets) · [GPT-2 Download](#-offline-gpt-2-checkpoint) · [Structure](#-repository-structure) · [Requirements](#-requirements) · [Usage](#-quick-start) · [Reproducibility](#-reproducibility-notes)


> 📦 **Lightweight GitHub release:** This repository provides the source code and instructions only. The large `datasets/` and `gpt2_offline/` directories are **not included in GitHub**. Download them separately following the instructions below before running any experiments.

## 🌟 Main Contributions

- **🔎 Window-scoped query contract.** Characterizes the inputs and identities of derived `Segment`, `Affinity`, and `Member` relations, including source revision, time interval, selected channels, normalization, and construction settings.
- **🕸️ Explicit group semantics.** Separates trend-based pair scoring, mutual-neighbor selection, and group enumeration. The paper shows why the same pair support need not imply the same hypergraph incidence relation and analyzes the work associated with relation construction.
- **🧪 Separated empirical evidence.** Connects the archived GHEF forecasting results with independent finite-graph checks and SQLite experiments on synthetic audit records, without treating those checks as end-to-end GHEF query acceleration.

The reported forecasting study covers **seven benchmark datasets** and **four prediction horizons**. Relative to CVC, the archived results show lower MAE in **22 of 28** dataset–horizon comparisons, with one tie and five losses. These are **reported paper results**, not newly reproduced measurements from this repository.

## 🧩 Method Overview

The GHEF forecasting pipeline has three components:

- **📉 GSA:** Represents local trends with multi-trend fuzzy information granules (MT-FIGs), including slope, intercept, fluctuation, and duration.
- **🔗 HDL:** Builds channel-dependency representations from trend-based affinities and propagates features through a hypergraph.
- **🧠 HGF:** Fuses temporal and relational representations for multivariate forecasting, using an offline GPT-2 backbone and trainable adaptation components.

> **Scope:** The paper specifies logical trend/affinity/membership relations and discusses group-enumeration policies. The supplied forecasting code uses an approximate, binned association computation and a weighted per-channel incidence construction; it is **not** a native database engine or a complete implementation of the paper's independent CPU/SQLite audits.

## 📚 Datasets

Seven public multivariate time-series benchmarks are used in the paper. **Datasets are distributed separately via Google Drive, not included in this GitHub repository.** The forecasting code loads a CSV file for each dataset from `datasets/<dataset>/<dataset>.csv`.

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

### ☁️ Download the datasets (Google Drive)

**🔗 Google Drive download link:** `GOOGLE_DRIVE_LINK_TO_BE_ADDED`

> 📌 **Repository owner:** Replace `GOOGLE_DRIVE_LINK_TO_BE_ADDED` with the Google Drive sharing URL before publishing this README. The datasets are **not** in GitHub because of their size.

1. 📥 Open the Google Drive link above and download the dataset archive or folder.
2. 📂 Extract its contents into a folder named `datasets/` at the **repository root** (alongside `code/` and `requirements.txt`).
3. ✅ Confirm that the directory layout matches the filenames expected by the scripts, for example:

```text
📊 datasets/
├── 🌡️ ETTh1/           └── 📄 ETTh1.csv
├── 🌡️ ETTh2/           └── 📄 ETTh2.csv
├── ⏱️ ETTm1/           └── 📄 ETTm1.csv
├── ⏱️ ETTm2/           └── 📄 ETTm2.csv
├── 🌦️ weather/         └── 📄 weather.csv
├── ⚡ electricity/     └── 📄 electricity.csv
└── 🚦 traffic/         └── 📄 traffic.csv
```

The input loader expects the dataset folder and CSV basename to match exactly, including capitalization (e.g., `datasets/ETTh1/ETTh1.csv`). If the downloaded archive contains an extra outer folder, move its contents so these paths exist.

## 🤗 Offline GPT-2 Checkpoint

The GHEF code loads GPT-2 from a **local checkpoint folder**. Because the pretrained weights are too large for this GitHub release, **download the six files manually from Hugging Face** and save them inside `gpt2_offline/`.

**🌐 Official model files:** **[openai-community/gpt2 — Hugging Face](https://huggingface.co/openai-community/gpt2/tree/main)**

### 📥 Files to download

Click a file below, then select **Download** on its Hugging Face page. Keep the original filenames.

| 📁 File | 📝 Purpose | 🔗 Download page |
|:--|:--|:--|
| ⚙️ `config.json` | GPT-2 architecture settings | [config.json](https://huggingface.co/openai-community/gpt2/blob/main/config.json) |
| 🧩 `merges.txt` | Byte-pair encoding merge rules | [merges.txt](https://huggingface.co/openai-community/gpt2/blob/main/merges.txt) |
| 🧠 `model.safetensors` | Pretrained GPT-2 model weights | [model.safetensors](https://huggingface.co/openai-community/gpt2/blob/main/model.safetensors) |
| 🔤 `tokenizer.json` | Fast-tokenizer definition | [tokenizer.json](https://huggingface.co/openai-community/gpt2/blob/main/tokenizer.json) |
| 🔧 `tokenizer_config.json` | Tokenizer configuration | [tokenizer_config.json](https://huggingface.co/openai-community/gpt2/blob/main/tokenizer_config.json) |
| 🔠 `vocab.json` | GPT-2 tokenizer vocabulary | [vocab.json](https://huggingface.co/openai-community/gpt2/blob/main/vocab.json) |

After downloading, your **local** folder must look like this:

```text
🤗 gpt2_offline/
├── ⚙️ config.json
├── 🧩 merges.txt
├── 🧠 model.safetensors
├── 🔤 tokenizer.json
├── 🔧 tokenizer_config.json
└── 🔠 vocab.json
```

> ✅ **Important:** `model.safetensors` must be the actual downloaded model file, not a Git LFS/Xet pointer or an HTML page. You do not need to download `pytorch_model.bin` in addition to `model.safetensors`. The current code uses `GPT2Model.from_pretrained(..., local_files_only=True)`, so it **will not download missing files automatically**. The publicly hosted GPT-2 checkpoint is a practical default; exact reproduction of archived experiments additionally depends on matching their original checkpoint and configuration.

## 🗂️ Repository Structure

The **GitHub checkout** contains the Python implementation and documentation. The two large directories shown below are created **locally after download**, not tracked in Git.

```text
GHEF/
├── 📘 README.md                                # Documentation and download instructions
├── 📦 requirements.txt                         # Python package requirements
│
├── 💻 code/                                    # Tracked GHEF source files
│   ├── 🚀 main.py                              # Training and evaluation entry point
│   ├── ⚙️ config.py                            # Hyperparameters and settings
│   ├── 📥 data_utils.py                        # CSV loading and normalization
│   ├── 🧠 model_ghef.py                        # GHEF model and offline GPT-2 backbone
│   ├── 📈 mtfig.py                             # Trend granulation and embeddings
│   ├── 🕸️ mic_hypergraph.py                    # Affinity scoring and incidence
│   ├── 🔗 hypergraph_layers.py                 # Hypergraph propagation layers
│   ├── 🎯 losses.py                            # Forecasting and auxiliary losses
│   ├── 🏋️ trainer.py                           # Multi-stage training / evaluation
│   └── 🧱 precompute_granules.py               # Optional granule precomputation
│
├── 📊 datasets/                                # 📥 DOWNLOAD FROM GOOGLE DRIVE (local only)
│   ├── 🌡️ ETTh1/                               # 📄 ETTh1.csv
│   ├── 🌡️ ETTh2/                               # 📄 ETTh2.csv
│   ├── ⏱️ ETTm1/                               # 📄 ETTm1.csv
│   ├── ⏱️ ETTm2/                               # 📄 ETTm2.csv
│   ├── 🌦️ weather/                             # 📄 weather.csv
│   ├── ⚡ electricity/                         # 📄 electricity.csv
│   └── 🚦 traffic/                             # 📄 traffic.csv
│
├── 🤗 gpt2_offline/                            # 📥 DOWNLOAD FROM HUGGING FACE (local only)
│   ├── ⚙️ config.json                          # GPT-2 configuration
│   ├── 🧩 merges.txt                           # Tokenizer BPE merges
│   ├── 🧠 model.safetensors                    # Pretrained weights
│   ├── 🔤 tokenizer.json                       # Tokenizer definition
│   ├── 🔧 tokenizer_config.json                # Tokenizer settings
│   └── 🔠 vocab.json                           # Tokenizer vocabulary
│
└── 📂 outputs/                                 # Generated experiment outputs (local only)
```

> 📝 **GitHub vs local:** `code/`, `README.md`, `requirements.txt`, and `.gitignore` are committed. `datasets/`, `gpt2_offline/`, and `outputs/` exist on the researcher's machine after setup and are excluded from version control. All commands below assume they are run from the project root.

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

### 1. 📥 Get the repository

Clone the GitHub repository (replace the placeholder with your actual repository URL), or download its source-code ZIP:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd GHEF
```

### 2. ⚙️ Install dependencies

From the `GHEF/` root directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 📊 Download the datasets from Google Drive

Follow [**dataset download instructions**](#-datasets) above. Place all seven benchmark CSV files under their matching `datasets/<dataset>/` subfolders.

### 4. 🤗 Download the offline GPT-2 files from Hugging Face

Follow [**🤗 Offline GPT-2 Checkpoint**](#-offline-gpt-2-checkpoint) above. Download the six files listed there and save them directly inside `gpt2_offline/`. Do **not** nest another `gpt2/` folder inside it.

Before running, you can check that the local checkpoint can be loaded:

```bash
python -c "from transformers import GPT2Model; GPT2Model.from_pretrained('./gpt2_offline', local_files_only=True); print('Local GPT-2 ready')"
```

### 5. ▶️ Run one dataset and one horizon

Run the following command **from the repository root**:

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

For CPU-only use, replace `--device cuda` with `--device cpu` (training will be substantially slower), and consider `--num_workers 0 --no_amp`.

### 6. 🧪 Run all seven datasets and four horizons

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

These commands execute the released code, but they do **not** guarantee exact reproduction of the paper's archived measurements. In particular, the current code contains dataset-level overrides; see [Reproducibility Notes](#-reproducibility-notes).

### 7. 🔧 Common command-line arguments

| ⚙️ Argument | 📝 Meaning | 💡 Code default |
|:--|:--|:--|
| `--datasets` | One or more benchmark names | All seven benchmarks |
| `--horizons` | Forecast lengths | `96 192 336 720` |
| `--lookback` | Historical sequence length | `512` |
| `--batch` | Batch size | `32` |
| `--lr` | Learning rate | `2e-4` |
| `--epochs` | Maximum Stage-3 epochs | `150` |
| `--stage1_epochs` | Stage-1 epochs | `20` |
| `--stage2_epochs` | Stage-2 epochs | `20` |
| `--patience` | Early-stopping patience | `10` |
| `--topk` | Neighbor-selection budget | `16` |
| `--device` | Device | `cuda` |
| `--no_stage1` / `--no_stage2` / `--no_stage3` | Disable selected training stages | All enabled |

For settings not exposed on the command line, see [`code/config.py`](code/config.py). Disabling a stage does **not** load a pretrained training-stage checkpoint automatically.

### 8. 🧱 Optional MT-FIG precomputation

```bash
python code/precompute_granules.py \
  --data_root ./datasets \
  --dataset ETTh1 \
  --out_dir ./granule_cache \
  --lookback 96 \
  --device cuda
```

The utility writes cached `.pt` files separately; the training script does not require the precomputed cache.

## 📁 Outputs and Evaluation

After running the main script, prediction errors and checkpoints are written under the selected `--out_dir` (default `./outputs`):

```text
📂 outputs/
├── 🌡️ ETTh1/
│   └── ⏳ H96/
│       ├── 💾 stage1.pt
│       ├── 💾 stage2.pt
│       └── 💾 stage3.pt
├── 📊 results_ETTh1.json
└── 📋 results_all.json
```

- 📉 **MAE** — Mean Absolute Error.
- 📊 **MSE** — Mean Squared Error.
- 💾 `results_<dataset>.json` stores per-horizon scores and their arithmetic average.
- 🗃️ `results_all.json` consolidates the evaluated datasets.
- 🧠 `stage*.pt` checkpoints store the selected validation-best model weights for each enabled training stage.

## 🔬 Reproducibility Notes

The paper and the present code snapshot are **not configuration-identical**. Before attempting a matched replication, check at least these differences:

1. **📏 Lookback:** The paper reports `96`; the code defaults to `512` and overrides the lookback to `192` for `electricity` and `96` for `traffic`.
2. **🧠 GPT-2 layers:** The paper describes the first **6** layers; `code/config.py` currently sets `gpt2_use_first_n_layers = 8`. This setting is not a CLI argument.
3. **🕒 Training budget:** The paper reports up to **100** epochs; the current Stage-3 code default is **150**, configurable via `--epochs`.
4. **📐 Non-ETT splitting:** The code uses a **70% / 10% / 20%** train/validation/test split for non-ETT datasets, whereas the paper's reported non-ETT split sizes correspond to approximately **70% / 20% / 10%**.
5. **🕸️ Group construction:** The current code computes a discretized, candidate-pruned association approximation and weighted incidence; it does not implement the full logical clique-enumeration policy analyzed in the paper.
6. **🗄️ Audit scope:** The independent finite-graph and SQLite audit programs referenced in the paper are **not included** in the supplied 10-file code archive.

Accordingly, the included commands document **how to run this released code**, while the archived tables in the paper should be read as separately reported evidence.
