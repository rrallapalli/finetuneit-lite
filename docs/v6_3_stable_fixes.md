# FineTuneIT Light v6.3 Stable Fixes

## Changes

- Removed TorchAO from the setup flow.
- Added defensive `pip uninstall -y torchao` after Unsloth installation.
- Added setup validation to confirm TorchAO is not installed.
- Added adapter-save verification after training.
- Added `uploaded_to_hub`, `hub_error`, and `output_files` to the training response.
- Added `scripts/inspect_outputs.sh`.

## Inspect saved adapters

```bash
bash scripts/inspect_outputs.sh
```

A successful LoRA save should include:

```text
adapter_config.json
adapter_model.safetensors
```
