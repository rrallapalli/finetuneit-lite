from pathlib import Path

import wandb
from huggingface_hub import HfApi
from transformers import TrainingArguments
from trl import SFTTrainer

from app.training.dataset_loader import load_and_prepare_dataset
from app.core.platform_config import get_wandb_project
from app.core.secrets import apply_secrets_to_environment, get_hf_token


def _load_unsloth():
    """Import Unsloth only when a training job starts."""
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        return FastLanguageModel, is_bfloat16_supported
    except Exception as exc:
        raise RuntimeError(
            "Unsloth could not be imported. On RunPod, run: "
            "`bash scripts/setup_runpod_gpu_env.sh`. If TorchAO appears in the "
            "traceback, run `pip uninstall -y torchao`. "
            f"Original error: {exc}"
        ) from exc


def _list_output_files(output_dir: str) -> list[str]:
    path = Path(output_dir)
    if not path.exists():
        return []
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def _has_adapter_weights(output_dir: str) -> bool:
    path = Path(output_dir)
    return (
        (path / "adapter_config.json").exists()
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def _save_adapter_artifacts(model, tokenizer, trainer, output_dir: str) -> list[str]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    save_errors = []

    try:
        trainer.model.save_pretrained(output_dir)
    except Exception as exc:
        save_errors.append(f"trainer.model.save_pretrained failed: {exc}")

    if not _has_adapter_weights(output_dir):
        try:
            model.save_pretrained(output_dir)
        except Exception as exc:
            save_errors.append(f"model.save_pretrained failed: {exc}")

    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:
        save_errors.append(f"tokenizer.save_pretrained failed: {exc}")

    output_files = _list_output_files(output_dir)

    if not _has_adapter_weights(output_dir):
        raise RuntimeError(
            "Training finished, but adapter artifacts were not saved correctly. "
            f"Output directory: {output_dir}. Files found: {output_files}. "
            f"Save errors: {save_errors}"
        )

    return output_files


def run_training_from_config(config: dict) -> dict:
    apply_secrets_to_environment()

    FastLanguageModel, is_bfloat16_supported = _load_unsloth()

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    lora_cfg = config.get("lora", {})
    output_config = config.get("output", {})

    experiment_name = config.get("experiment_name", "finetuneit_experiment")
    wandb_project = config.get("wandb_project", get_wandb_project())
    wandb_entity = config.get("wandb_entity") or None

    output_dir = output_config.get("output_dir", f"outputs/{experiment_name}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run = wandb.init(project=wandb_project, entity=wandb_entity, name=experiment_name, config=config)

    try:
        dataset = load_and_prepare_dataset(config)

        base_model = model_config.get("base_model", "unsloth/Qwen2.5-0.5B")
        max_seq_length = int(model_config.get("max_seq_length", 1024))
        load_in_4bit = bool(model_config.get("load_in_4bit", True))

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            token=get_hf_token(),
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=int(lora_cfg.get("r", 16)),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
            bias=lora_cfg.get("bias", "none"),
            use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
            random_state=int(training_config.get("seed", 3407)),
            use_rslora=bool(lora_cfg.get("use_rslora", False)),
            loftq_config=None,
        )

        bf16_supported = is_bfloat16_supported()

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
            warmup_steps=int(training_config.get("warmup_steps", 5)),
            max_steps=int(training_config.get("max_steps", 25)),
            learning_rate=float(training_config.get("learning_rate", 2e-4)),
            fp16=not bf16_supported,
            bf16=bf16_supported,
            logging_steps=int(training_config.get("logging_steps", 1)),
            optim=training_config.get("optim", "adamw_8bit"),
            weight_decay=float(training_config.get("weight_decay", 0.01)),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
            seed=int(training_config.get("seed", 3407)),
            report_to="wandb",
            eval_strategy="steps",
            eval_steps=int(training_config.get("eval_steps", 5)),
            save_steps=int(training_config.get("save_steps", 10)),
            save_total_limit=int(training_config.get("save_total_limit", 2)),
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=args,
            packing=False,
        )

        train_result = trainer.train()

        output_files = _save_adapter_artifacts(model, tokenizer, trainer, output_dir)

        adapter_repo = output_config.get("output_adapter_repo")
        push_to_hub = bool(output_config.get("push_to_hub", False))
        private_repo = bool(output_config.get("private_repo", False))

        uploaded_to_hub = False
        hub_error = None

        if push_to_hub and adapter_repo and not adapter_repo.startswith("your-hf-username"):
            try:
                api = HfApi(token=get_hf_token())
                api.create_repo(adapter_repo, exist_ok=True, private=private_repo)
                api.upload_folder(repo_id=adapter_repo, folder_path=output_dir, repo_type="model")
                uploaded_to_hub = True
            except Exception as exc:
                hub_error = str(exc)

        wandb.log({
            "adapter_saved": True,
            "uploaded_to_hub": uploaded_to_hub,
            "output_file_count": len(output_files),
        })

        result = {
            "status": "completed",
            "experiment_name": experiment_name,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_run_id": run.id,
            "wandb_run_url": run.url,
            "base_model": base_model,
            "adapter_repo": adapter_repo,
            "uploaded_to_hub": uploaded_to_hub,
            "hub_error": hub_error,
            "output_dir": output_dir,
            "output_files": output_files,
            "train_metrics": train_result.metrics,
        }

        wandb.finish()
        return result

    except Exception as exc:
        output_files = _list_output_files(output_dir)
        wandb.finish()
        return {
            "status": "error",
            "experiment_name": experiment_name,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_run_id": run.id if run else None,
            "wandb_run_url": run.url if run else None,
            "output_dir": output_dir,
            "output_files": output_files,
            "message": str(exc),
        }
