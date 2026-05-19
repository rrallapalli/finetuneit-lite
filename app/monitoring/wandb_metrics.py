import math
from typing import Any

import pandas as pd
import wandb

from app.core.secrets import apply_secrets_to_environment


def clean_json(obj: Any):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [clean_json(v) for v in obj]

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj


def fetch_wandb_run_summary(entity: str, project: str, run_id: str) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    result = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "summary": dict(run.summary),
        "config": dict(run.config),
    }

    return clean_json(result)


def fetch_wandb_history(entity: str, project: str, run_id: str, samples: int = 500) -> dict:
    apply_secrets_to_environment()
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    history = run.history(samples=samples)

    if history is None or history.empty:
        return {"columns": [], "records": []}

    history = history.replace([float("inf"), float("-inf")], None)
    history = history.where(pd.notnull(history), None)

    result = {
        "columns": list(history.columns),
        "records": history.to_dict(orient="records"),
    }

    return clean_json(result)
