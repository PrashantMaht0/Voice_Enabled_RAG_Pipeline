"""Builds the folder uploaded to the Hugging Face Space (spec 0009, AC-14): the app, its requirements, the shared
ingestion code it imports (src/ingestion, src/utils/config.py) and a config.yaml made of the repository's
config/config.yaml with the Space's overrides. The repository stays the only source; the Space is never edited.

Usage, from the repo root: python space/assemble.py [out_dir]   (default: space/.dist)
"""
import os
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPACE = os.path.join(ROOT, "space")
APP_FILES = ("app.py", "reader.py", "README.md", "requirements.txt", "packages.txt")
SHARED = ("src/ingestion", "src/utils/__init__.py", "src/utils/config.py")


def merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        base[key] = merge(base.get(key) or {}, value) if isinstance(value, dict) else value
    return base


def assemble(out: str) -> str:
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for name in APP_FILES:
        shutil.copy(os.path.join(SPACE, name), out)
    for path in SHARED:
        source, target = os.path.join(ROOT, path), os.path.join(out, path)
        if os.path.isdir(source):
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy(source, target)
    with open(os.path.join(ROOT, "config", "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(os.path.join(SPACE, "config.overrides.yaml"), encoding="utf-8") as f:
        overrides = yaml.safe_load(f)
    keep = {key: config[key] for key in ("ingestion",) if key in config}   # the reader needs nothing else
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(merge(keep, overrides), f, sort_keys=False)
    return out


if __name__ == "__main__":
    print(assemble(sys.argv[1] if len(sys.argv) > 1 else os.path.join(SPACE, ".dist")))
