"""配置加载：极简 YAML 子集（嵌套 map / 标量 / 内联列表）。"""
from __future__ import annotations

from pathlib import Path


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_scalar(s: str):
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p.strip()) for p in inner.split(",")]
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip('"').strip("'")


def load_config(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        indent = len(raw) - len(raw.lstrip())
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(val)
    return root


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
