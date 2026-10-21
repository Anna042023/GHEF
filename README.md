# 📈 Window-Scoped Trend Relations for Multivariate Time-Series Analytics

**GHEF — Granule-Hypergraph Enhanced Forecaster**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://huggingface.co/docs/transformers/)
[![Task](https://img.shields.io/badge/Task-Multivariate%20Forecasting-5B8DEF)](#-overview)

This repository contains the **GHEF forecasting implementation** associated with the paper **“Window-Scoped Trend Relations for Multivariate Time-Series Analytics.”** The paper studies how historical time windows define derived trend records, channel affinities, and group memberships, using GHEF as a forecasting consumer.

**🔗 Navigation:** [Contributions](#-main-contributions) · [Method](#-method-overview) · [Datasets](#-datasets) · [GPT-2 Download](#-offline-gpt-2-checkpoint) · [Structure](#repository-structure) · [Requirements](#requirements) · [Usage](#-quick-start)  · [Outputs](#outputs)


> 📦 **Lightweight GitHub release:** This repository provides the source code and instructions only. The large `datasets/` and `gpt2_offline/` directories are **not included in GitHub**. Download them separately following the instructions below before running any experiments.

## 🌟 Main Contributions

- **🔎 Window-scoped query contract.** Characterizes the inputs and identities of derived `Segment`, `Affinity`, and `Member` relations, including source revision, time interval, selected channels, normalization, and construction settings.
- **🕸️ Explicit group semantics.** Separates trend-based pair scoring, mutual-neighbor selection, and group enumeration. The paper shows why the same pair support need not imply the same hypergraph incidence relation and analyzes the work associated with relation construction.
- **🧪 Separated empirical evidence.** Connects the archived GHEF forecasting results with independent finite-graph checks and SQLite experiments on synthetic audit records, without treating those checks as end-to-end GHEF query acceleration.

The reported forecasting study covers **seven benchmark datasets** and **four prediction horizons**. Relative to CVC, the archived results show lower MAE in **22 of 28** dataset–horizon comparisons, with one tie and five losses.

## 🧩 Method Overview

The GHEF forecasting pipeline has three components:

- **📉 GSA:** Represents local trends with multi-trend fuzzy information granules (MT-FIGs), including slope, intercept, fluctuation, and duration.
- **🔗 HDL:** Builds channel-dependency representations from trend-based affinities and propagates features through a hypergraph.
- **🧠 HGF:** Fuses temporal and relational representations for multivariate forecasting, using an offline GPT-2 backbone and trainable adaptation components.
  
## 📚 Datasets

Seven public multivariate time-series benchmarks are used in the paper. **Datasets are distributed separately via Google Drive, not included in this GitHub repository.** 

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

### ☁️ Download the datasets

**🔗 Google Drive download link:**   [Google Drive](https://drive.google.com/file/d/1P6ZQY1U1NICWmlpXoDlK-fiu6Kjt_KRb/view?usp=sharing)

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

## 🤗 Offline GPT-2 Checkpoint

The GHEF code loads GPT-2 from a **local checkpoint folder**. **Download the six files manually from Hugging Face** and save them inside `gpt2_offline/`.

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

<a id="repository-structure"></a>

## 🗂️ Repository Structure

The **GitHub checkout** contains the Python implementation and documentation.

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

<a id="requirements"></a>

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

## 🚀 Quick Start

### 1. 📥 Get the repository

Clone the GitHub repository.

### 2. ⚙️ Install dependencies

From the `GHEF/` root directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 📊 Download the datasets

Follow [**dataset download instructions**](#-datasets) above. Place all seven benchmark CSV files under their matching `datasets/<dataset>/` subfolders.

### 4. 🤗 Download the offline GPT-2 files

Follow [**🤗 Offline GPT-2 Checkpoint**](#-offline-gpt-2-checkpoint) above. Download the six files listed there and save them directly inside `gpt2_offline/`. Do **not** nest another `gpt2/` folder inside it.

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

### 7. 🧱 Optional MT-FIG precomputation

```bash
python code/precompute_granules.py \
  --data_root ./datasets \
  --dataset ETTh1 \
  --out_dir ./granule_cache \
  --lookback 96 \
  --device cuda
```

<a id="outputs"></a>

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
