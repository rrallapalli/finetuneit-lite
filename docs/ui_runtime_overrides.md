# UI Runtime Overrides

FineTuneIT Lite v6 adds user-facing runtime inputs for W&B and Hugging Face artifact settings.

## What moved to the UI

The Training tab now allows the user to enter:

```text
W&B Entity
W&B Project
Hugging Face Username / Org
Adapter Repo Name
Push Adapter to Hugging Face
Private HF Repo
```

The app constructs the adapter repository automatically:

```text
<HF username>/<adapter repo name>
```

Example:

```text
rrallapalli/qwen25-reasoning-lora
```

## What stays in secrets

Tokens remain outside the UI and are loaded from:

```text
configs/secrets/secrets.yaml
```

```yaml
huggingface:
  token: hf_xxxxx

wandb:
  api_key: xxxxx

runpod:
  api_key: xxxxx
```

## Why this design

This creates a cleaner platform pattern:

```text
Secrets Layer
  → credentials only

Platform Config Layer
  → defaults and infrastructure behavior

UI Runtime Overrides
  → user-specific experiment settings
```



## Latest Training Context Reuse

The app stores the runtime values entered during training in Streamlit session state:

```text
W&B Entity
W&B Project
HF Username
Adapter Repo
Base Model
Prompt Template
Dataset Path
W&B Run ID, when available
```

These values are reused automatically in:

```text
Live Metrics
Experiment Registry
Evaluate
Inference Playground
```

This reduces repeated manual entry after submitting a training run.
