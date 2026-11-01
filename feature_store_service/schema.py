"""Small deterministic validator; schemas are data, not executable code."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


class SchemaRegistry:
    def __init__(self, path=None):
        path = Path(path or Path(__file__).with_name("schemas.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or not isinstance(value.get("schemas"), dict):
            raise ValueError("unsupported feature schemas")
        self.schemas = value["schemas"]

    def list(self):
        return {"schema_version": 1, "schemas": [
            {"name": name, "fields": sorted(schema["fields"]),
             "required": list(schema.get("required", [])), "task":schema["task"]}
            for name, schema in sorted(self.schemas.items())
        ]}

    def task(self,name):
        schema=self.schemas.get(name)
        if schema is None: raise ValueError("unknown feature schema")
        return schema["task"]

    def validate(self, name, values):
        schema = self.schemas.get(name)
        if schema is None:
            raise ValueError("unknown feature schema")
        if not isinstance(values, dict) or set(values) - set(schema["fields"]):
            raise ValueError("unknown feature fields")
        if not set(schema.get("required", [])) <= set(values):
            raise ValueError("missing feature fields")
        for field, value in values.items():
            rule = schema["fields"][field]
            kind = rule["type"]
            if kind == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{field} must be finite number")
                if not rule["minimum"] <= value <= rule["maximum"]:
                    raise ValueError(f"{field} out of range")
            elif kind == "boolean":
                if not isinstance(value, bool):
                    raise ValueError(f"{field} must be boolean")
            elif kind == "count_map":
                if not isinstance(value, dict) or len(value) > rule["maximum_items"]:
                    raise ValueError(f"{field} invalid count map")
                if any(not isinstance(key, str) or not KEY.fullmatch(key)
                       or isinstance(count, bool) or not isinstance(count, int)
                       or not 0 <= count <= rule["maximum"] for key, count in value.items()):
                    raise ValueError(f"{field} invalid count map")
            else:
                raise ValueError("unsupported field type")
        return json.loads(json.dumps(values, ensure_ascii=False, separators=(",", ":")))
