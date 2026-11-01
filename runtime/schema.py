"""最小 Schema Validation — Capability Contract v0.1.

只支持 JSON-schema 的一个子集:
  type: object/string/integer/number/boolean/array/null
  object: properties, required, additionalProperties(false)
  integer/number: minimum, maximum
  string/integer: enum
  array: items

完整 JSON Schema 支持不在 v0.1 范围。
"""

from .errors import SchemaValidationError


def validate(schema, value, path="$"):
    """校验 value 是否符合 schema; 不符则抛 SchemaValidationError。"""
    if schema is None:
        return

    t = schema.get("type")

    if t == "object":
        _validate_object(schema, value, path)
    elif t == "string":
        if not isinstance(value, str):
            _fail(path, f"期望 string, 实际 {type(value).__name__}")
    elif t == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(path, f"期望 integer, 实际 {type(value).__name__}")
        _check_range(schema, value, path)
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(path, f"期望 number, 实际 {type(value).__name__}")
        _check_range(schema, value, path)
    elif t == "boolean":
        if not isinstance(value, bool):
            _fail(path, f"期望 boolean, 实际 {type(value).__name__}")
    elif t == "array":
        if not isinstance(value, (list, tuple)):
            _fail(path, f"期望 array, 实际 {type(value).__name__}")
        if "minItems" in schema and len(value) < schema["minItems"]:
            _fail(path, f"数组长度 {len(value)} < minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            _fail(path, f"数组长度 {len(value)} > maxItems {schema['maxItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                validate(item_schema, item, f"{path}[{i}]")
    elif t == "null":
        if value is not None:
            _fail(path, f"期望 null, 实际 {type(value).__name__}")
    # 未声明 type: 跳过类型检查

    if "enum" in schema and value not in schema["enum"]:
        _fail(path, f"值 {value!r} 不在枚举 {schema['enum']} 中")


def _validate_object(schema, value, path):
    if not isinstance(value, dict):
        _fail(path, f"期望 object, 实际 {type(value).__name__}")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for k in required:
        if k not in value:
            _fail(path, f"缺少必填字段 {k!r}")

    for k, sub in properties.items():
        if k in value:
            validate(sub, value[k], f"{path}.{k}")

    if schema.get("additionalProperties") is False:
        for k in value:
            if k not in properties:
                _fail(path, f"不允许的额外字段 {k!r}")


def _check_range(schema, value, path):
    if "minimum" in schema and value < schema["minimum"]:
        _fail(path, f"{value} < minimum {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        _fail(path, f"{value} > maximum {schema['maximum']}")


def _fail(path, msg):
    raise SchemaValidationError(f"{path}: {msg}", details={"path": path})
