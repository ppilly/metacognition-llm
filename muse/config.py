"""YAML config loading with `extends:` inheritance, ${...} interpolation and dotted CLI overrides."""
import copy
import os
import re
from pathlib import Path

import yaml


class Config(dict):
    """dict with attribute access (cfg.model.name)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __deepcopy__(self, memo):
        return _wrap(copy.deepcopy(dict(self), memo))


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _merge(_load_yaml(path.parent / parent), data)
    return data


_VAR = re.compile(r"\$\{([^}]+)\}")


def _lookup(root: dict, dotted: str):
    node = root
    for part in dotted.split("."):
        node = node[part]
    return node


def _interpolate(node, root):
    if isinstance(node, dict):
        return {k: _interpolate(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root) for v in node]
    if isinstance(node, str):
        def sub(m):
            key = m.group(1)
            if "." in key:
                return str(_lookup(root, key))
            # unresolved env vars are left in place; consumers raise if they actually need them
            return os.environ.get(key, m.group(0))
        return _VAR.sub(sub, node)
    return node


def _parse_value(text: str):
    return yaml.safe_load(text)


def load_config(path: str, overrides: list[str] | None = None) -> Config:
    data = _load_yaml(Path(path))
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    # interpolate twice so ${run.name} inside run.out_dir resolves after overrides
    data = _interpolate(_interpolate(data, data), data)
    return _wrap(data)
