# FineTuneIT v6.2 Stable — RunPod Instructions

This version intentionally keeps the platform simple and stable.

## Included

- Streamlit UI
- FastAPI backend
- W&B/HF UI inputs
- W&B live metrics
- Experiment registry
- Evaluation
- Inference playground
- NaN-safe W&B metrics
- Lazy Unsloth import
- Simple RunPod scripts

## Removed for stability

- Project-wide v7 dashboard
- Bootstrap automation
- Multi-run comparison
- torchao installation

## RunPod flow

```bash
git clone https://github.com/rrallapalli/finetuneit-lite.git
cd finetuneit-lite
chmod +x scripts/*.sh
bash scripts/setup_runpod_gpu_env.sh
bash scripts/configure_secrets.sh
```

Terminal 1:

```bash
bash scripts/start_backend.sh
```

Terminal 2:

```bash
bash scripts/start_streamlit.sh
```

Open RunPod HTTP port `8501`.

## Recommended first test

Use:

```text
Model: unsloth/Qwen2.5-0.5B
Dataset: data/sample_alpaca.jsonl
Max steps: 25
Logging steps: 1
Eval steps: 5
Batch size: 1
```
