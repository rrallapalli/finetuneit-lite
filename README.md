# FineTuneIT Lite

A lightweight, RunPod-ready fine-tuning platform prototype with a single Streamlit product app:

```text
Train → Live Metrics → Evaluate → Inference Playground
```

This version includes W&B live metric polling, local/runpod execution routing, and RunPod Pod helper scripts.

---

# FineTuneIT Lite

A lightweight, RunPod-ready fine-tuning platform prototype for domain-agnostic LLM adaptation.

This version supports **configurable fine-tuning experiments** from the UI:

- Reasoning vs non-reasoning dataset modes
- Hugging Face dataset or local JSONL dataset
- Alpaca and reasoning prompt templates
- Base model selection
- LoRA hyperparameters
- Training hyperparameters
- W&B experiment tracking
- Hugging Face LoRA adapter publishing
- Post-training model evaluation
- RunPod training, evaluation, and inference workers

## Platform workflow

```text
User Configures Experiment
  ↓
Dataset Mode + Prompt Template Selection
  ↓
RunPod Training Job
  ↓
W&B Tracking
  ↓
LoRA Adapter Push to Hugging Face
  ↓
Evaluation Job
  ↓
Inference Worker
```

## Why this matters

The original capstone workflow fine-tuned models manually through notebooks. This project turns that into a configurable platform product where users can reproduce different experiments by changing parameters from the UI instead of editing notebook code.

## Supported experiment profiles

| Profile | Dataset Type | Example Dataset | Prompt Template | Purpose |
|---|---|---|---|---|
| Non-reasoning | Instruction tuning | `tatsu-lab/alpaca` | Alpaca | General instruction-following |
| Reasoning | Chain-of-thought style | `FreedomIntelligence/medical-o1-reasoning-SFT` | Reasoning Alpaca | Reasoning-style supervised fine-tuning |
| Custom JSONL | User uploaded | local `.jsonl` | Alpaca or reasoning | Portfolio/demo extension |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.api.main:app --reload
```

Run UI:

```bash
streamlit run app/ui/streamlit_app.py
```

## Main folders

```text
app/api/             FastAPI control plane
app/ui/              Streamlit configurable UI
app/training/        Training orchestration and config loading
app/evaluation/      Evaluation scripts and metrics
app/inference/       Adapter-based inference
configs/             YAML experiment profiles
runpod/              RunPod handlers and Dockerfiles
docs/                Architecture and configuration docs
data/                Sample datasets
```


## New in v3

- Added hyperparameter help pop-ups in the Streamlit training UI
- Added Experiment Registry tab
- Added W&B run listing and comparison
- Added model ranking score
- Added champion model promotion
- Added option to use champion model directly in Inference Playground



## New in v4

- Added `configs/platform_config.yaml`
- Moved execution mode, W&B defaults, RunPod endpoints, registry settings, UI refresh defaults, and inference defaults into YAML config
- Kept `.env` only for secrets: `HF_TOKEN`, `WANDB_API_KEY`, `RUNPOD_API_KEY`
- Added `app/core/platform_config.py`
- Updated API routes and Streamlit defaults to read from platform config



## New in v5

- Added `configs/secrets/secrets.example.yaml`
- Added local `configs/secrets/secrets.yaml` support
- Added `app/core/secrets.py`
- Added environment-variable fallback for secrets
- Updated Hugging Face, W&B, and RunPod access to use the secrets loader
- Added `.gitignore` protection for local secrets
- Added `docs/secrets_management.md`

Secrets now live outside platform and experiment configs:

```text
configs/platform_config.yaml          # runtime behavior
configs/*.yaml                        # experiment configs
configs/secrets/secrets.yaml          # local credentials, not committed
```



## New in v6

- Added W&B entity and project inputs directly in the Streamlit Training tab
- Added Hugging Face username / organization input
- Added adapter repository name input
- Added automatic adapter repo construction: `<hf_username>/<adapter_name>`
- Added push-to-Hugging Face toggle
- Added private Hugging Face repo toggle
- Kept tokens outside the UI using `configs/secrets/secrets.yaml`
- Added `docs/ui_runtime_overrides.md`



## v6 Enhancement: Runtime Context Reuse

The Streamlit app now reuses details entered during training across downstream workflows:

- Live Metrics auto-fills W&B entity/project and run ID when available
- Experiment Registry auto-fills W&B entity/project
- Evaluation auto-fills base model, adapter repo, and dataset path
- Inference auto-fills adapter repo, base model, and prompt template


## v6.2 Stable

This version focuses on reliability and a clean portfolio-ready workflow.

### Included

- Configurable fine-tuning UI
- W&B and Hugging Face inputs from the frontend
- W&B live metrics
- Experiment registry
- Evaluation
- Inference playground
- NaN-safe W&B metric handling
- Lazy Unsloth import
- Simple RunPod startup scripts

### Removed for stability

- v7 project-wide dashboard
- multi-run comparison
- bootstrap automation
- torchao dependency

### Recommended RunPod flow

```bash
git clone https://github.com/rrallapalli/finetuneit-lite.git
cd finetuneit-lite
chmod +x scripts/*.sh
bash scripts/setup_runpod_gpu_env.sh
bash scripts/setup_tokens.sh
```

Terminal 1:

```bash
bash scripts/start_backend.sh
```

Terminal 2:

```bash
bash scripts/start_streamlit.sh
```


## v6.3 Stable

This release fixes RunPod setup and adapter export reliability.

### Fixes

- Removed TorchAO from the setup flow
- Added defensive TorchAO uninstall after dependency installation
- Added adapter save verification after training
- Added explicit `uploaded_to_hub`, `hub_error`, and `output_files` in training response
- Added `scripts/inspect_outputs.sh`
