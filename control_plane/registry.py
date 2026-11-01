"""Validated read-only model registry with board-local artifact probes."""

from __future__ import annotations

import json
import re
from pathlib import Path


MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,95}$")


class ModelRegistry:
    def __init__(self, path=None):
        self.path = Path(path or Path(__file__).with_name("models.json"))
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or not isinstance(value.get("models"), list):
            raise ValueError("unsupported model registry")
        self.models = {}
        for model in value["models"]:
            model_id = model.get("id")
            if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id) or model_id in self.models:
                raise ValueError("invalid model id")
            if not isinstance(model.get("resources"), dict):
                raise ValueError("invalid model resources")
            self.models[model_id] = dict(model)

    def list(self, *, task=None):
        values = [self._status(value) for value in self.models.values()
                  if task is None or value.get("task") == task]
        return {"schema_version": 1, "models": values}

    def get(self, *, model_id):
        try:
            return self._status(self.models[model_id])
        except KeyError as exc:
            raise ValueError("model not found") from exc

    @staticmethod
    def _status(model):
        value = dict(model)
        artifacts = [Path(path).expanduser() for path in value.get("artifacts", [])]
        value["artifacts_ready"] = all(path.is_file() for path in artifacts) if artifacts else None
        value["artifact_count"] = len(artifacts)
        value.pop("artifacts", None)
        return value
