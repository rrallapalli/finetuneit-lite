import runpod
from app.evaluation.evaluate_model import evaluate_model


def handler(job):
    payload = job["input"]

    return evaluate_model(
        base_model=payload["base_model"],
        adapter_repo=payload.get("adapter_repo"),
        dataset_path=payload["dataset"],
        num_samples=payload.get("num_samples", 25),
        output_path=payload.get("output_path", "outputs/evaluation_metrics.csv"),
        wandb_project=payload.get("wandb_project", "finetuneit-lite"),
        job_id=payload.get("job_id", "eval-job"),
        prompt_template=payload.get("prompt_template", "alpaca"),
        max_new_tokens=payload.get("max_new_tokens", 256),
    )


runpod.serverless.start({"handler": handler})
