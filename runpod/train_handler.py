import runpod
from app.training.train_lora import run_training_from_config


def handler(job):
    payload = job["input"]

    if "config" in payload:
        config = payload["config"]
    else:
        config = payload

    return run_training_from_config(config)


runpod.serverless.start({"handler": handler})
