# 🗃️ Rank-Sensitive Top-K Hypergraph Views for Batch-Aware Multivariate Time-Series Analytics

**GHEF — Granule-Hypergraph Enhanced Forecaster** · Downstream forecasting workload

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://huggingface.co/docs/transformers/)
[![Research](https://img.shields.io/badge/Research-Time--Series%20Data%20Management-527A9B)](#-main-contributions)

This repository provides the **GHEF forecasting code** associated with the paper **“Rank-Sensitive Top-K Hypergraph Views for Batch-Aware Multivariate Time-Series Analytics.”** The paper investigates how to validate and maintain derived relations in batch-ranked multivariate time-series analytics. Its database focus is on **materialized Trend, Affinity, and Member views**, their reuse conditions, and the cost of refreshing them after source or batch revisions. GHEF serves as the downstream application that consumes these relations for forecasting.

**🔗 Navigation:** [Contributions](#-main-contributions) · [View Semantics](#-materialized-view-semantics) · [GHEF Workload](#-ghef-forecasting-workload) · [Evidence](#-experimental-evidence) · [Datasets](#-datasets) · [GPT-2 Download](#-offline-gpt-2-checkpoint) · [Structure](#repository-structure) · [Requirements](#requirements) · [Usage](#-quick-start) · [Outputs](#outputs)

> 📦 **Repository scope:** This release contains the GHEF Python forecasting implementation and its setup instructions. The paper's independent **SQLite view-validation, Top-K witness, and refresh-cost experiments are not included in this archive**. The datasets and local GPT-2 checkpoint are distributed separately.

## 🌟 Main Contributions

- **🔎 Stage-specific reuse conditions.** Defines separate identities and validity requirements for fitted `Trend`, batch-ranked `Affinity`, and weighted `Member` views. Reusing an Affinity score requires agreement of the ordered rank-bin inputs under fixed estimation and numerical rules. Reusing Member results additionally depends on candidate domains, tie decisions, Softmax weights, and write order.
- **🕸️ Joint Top-K consistency.** Distinguishes individually possible memberships from a globally realizable relation. For fixed scores and prescribed choices within boundary ties, selected-before-rejected precedence constraints admit a shared tie priority exactly when their directed graph is acyclic. Weighted results require further checks of normalization and ordered writes.
- **⚙️ Cost-aware incremental refresh.** Compares validation, cached reads, recomputation, and writes with full reconstruction after batch revisions. The analysis identifies cases where rank changes make selective refresh ineffective and cases where sparse changes allow substantial savings.

## 🔍 Materialized View Semantics

The paper models a batch-aware analytical pipeline with three derived relations. Each view has a distinct identity and reuse boundary.

| 🧱 View | 📄 Derived result | 🔐 Relevant dependencies |
|:--|:--|:--|
| **Trend** | Fitted temporal descriptors and MT-FIG records | Source revision, window or batch identity, alignment, normalization, fitting scope, channel, and segment-local origin |
| **Affinity** | Batch-ranked cross-channel scores | Trend inputs, candidate screening, ordered rank bins, fixed 16-bin NMI estimator, and numerical/symmetrization policy |
| **Member** | Selected Top-K neighbors and weighted hypergraph incidence | Affinity scores, full candidate domain, Top-K and tie policy, Softmax normalization, selection order, and symmetric writes |

A neighboring batch replacement can change the rank encoding used by an unchanged channel pair. The original pair values may agree while its cached Affinity score no longer satisfies the reuse contract. Equal pair scores also do not by themselves certify an identical weighted Member relation.

**🔄 Refresh decision:** check stage identities and consumed rank codes, validate the affected Top-K selections and weighted writes, then compare the cost of selective maintenance against rebuilding the required view. The relevant dependencies depend on whether the query requests scores, membership support, or final stored weights.

## 🧩 GHEF Forecasting Workload

The forecasting consumer follows the three-stage GHEF pipeline shown in the paper's Figure 2:

- **📉 GSA — Granulation and Adaptation:** Converts normalized histories into multi-trend fuzzy information granules (MT-FIGs), with slope, intercept, dispersion, and duration, and aligns the trend representation with a pretrained GPT-2 model.
- **🕸️ HDL — Hypergraph Dependency Learning:** Builds channel affinities, performs center-wise Top-K selection, and constructs a weighted incidence representation for hypergraph feature propagation.
- **🔀 HGF — Hypergraph-Guided Fusion:** Combines trend and hypergraph representations through bidirectional cross-attention and reconstructs future values from predicted granule parameters.

The forecasting formulation also uses an MIC-based affinity prior. **The fixed 16-bin rank-NMI scores evaluated in the SQLite view experiments are a separate, explicitly defined database scoring workload**. Their agreement and reuse results should not be interpreted as end-to-end forecasting speedups.

## 🧪 Experimental Evidence

The paper evaluates database-view correctness and maintenance separately from GHEF forecasting quality.

| 🔬 Evaluation | 🧾 Setup | 📊 Reported finding |
|:--|:--|:--|
| **Score and Member agreement (RQ1)** | Frozen 7-channel ETTh1 and 64-channel Electricity excerpts, with shared encoded inputs and selection policies | Maximum SQL-to-reference score differences are **1.19 × 10⁻⁷** and **2.38 × 10⁻⁷**, respectively. Shared selections produce matching Member hashes. |
| **Batch revision and refresh (RQ2)** | Electricity replacement over 64 channels | **2,010 / 2,016** candidate pairs are flagged for reassessment. Selective refresh has no consistent latency advantage in this near-global-change case. |
| **Sparse revision (RQ2)** | Constructed 64-channel input with four changed channels | Median selective-refresh time is **20.8 ms**, compared with **131.6 ms** for a full refresh under the reported matched protocol. |
| **Query access** | SQLite indexed lookup on synthetic graph summaries | A selective range query shows a **35.72×** scan-to-index speed ratio. The benefit decreases as the number of returned rows grows. |

**📈 Downstream forecasting (RQ3).** GHEF is evaluated on seven multivariate benchmarks at horizons **96, 192, 336, and 720**, with MAE and MSE. Against CVC, the paper reports the following counts of cases with lower, tied, or higher errors:

| Protocol | Cases | MAE (lower / tied / higher) | MSE (lower / tied / higher) |
|:--|--:|:--:|:--:|
| Standard forecasting | 28 | 22 / 1 / 5 | 18 / 0 / 10 |
| 10% training data | 28 | 25 / 0 / 3 | 23 / 1 / 4 |
| Zero-shot transfer | 32 | 24 / 2 / 6 | 21 / 1 / 10 |

These forecasting results characterize the GHEF consumer. **The `code/main.py` entry point supports the standard forecasting runs. Dedicated 10%-training, zero-shot-transfer, and SQLite audit runners are not present in this archive.**

## 📚 Datasets

The forecasting study uses seven public multivariate time-series benchmarks. Fixed excerpts from **ETTh1** and **Electricity** also support the paper's separate SQLite database-view evaluation. **The full benchmark datasets are distributed via the same Google Drive link as the earlier release and are not included in this GitHub archive.**

| 📁 Dataset | 🏷️ Domain | ⏱️ Sampling interval | 🔢 Variables |
|:--|:--|:--|--:|
| `ETTh1` | Transformer temperature | 1 hour | 7 |
| `ETTh2` | Transformer temperature | 1 hour | 7 |
| `ETTm1` | Transformer temperature | 15 minutes | 7 |
| `ETTm2` | Transformer temperature | 15 minutes | 7 |
| `weather` | Meteorological observations | 10 minutes | 21 |
| `electricity` | Electricity consumption | 1 hour | 321 |
| `traffic` | Road traffic occupancy | 1 hour | 862 |

**🎯 Forecast protocol reported in the paper:** lookback length **96** and prediction horizons **96 / 192 / 336 / 720**. Error metrics are **MAE** and **MSE** (lower is better). The forecasting evaluation is distinct from the SQLite view-maintenance measurements.

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

The **GitHub checkout** contains the forecasting Python implementation and documentation. Local dataset, model-checkpoint, and output directories retain the same layout as the earlier README.

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

The following commands execute **GHEF forecasting**, using the files already present in this repository. They do not reproduce the paper's independent SQLite validation and refresh benchmarks.

**⚠️ Configuration alignment:** The paper reports a **96-step lookback** and **six GPT-2 layers**, while the released `code/config.py` defaults to **eight GPT-2 layers** and applies a **192-step lookback override for Electricity**. For an exact match to the paper's forecasting protocol, check these settings and the archived experiment configurations before comparing numerical results. No source-code or directory changes are made in this README update.

### 1. 📥 Get the repository

Clone the GitHub repository:

```bash
git clone https://github.com/Anna042023/GHEF.git
cd GHEF
```

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

After running the **forecasting** entry point, prediction errors and checkpoints are written under the selected `--out_dir` (default `./outputs`):

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
