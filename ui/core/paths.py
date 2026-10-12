"""Filesystem locations shared by the UI.

The UI runs in its own venv (ui/.venv) and drives the animatediff CLI that lives
in the repo's main venv (venv/). Everything here is resolved relative to the
repo root so the UI works no matter which directory it was launched from.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
PRESET_DIR = CONFIG_DIR / "prompts"
UI_DIR = REPO_ROOT / "ui"
UI_STATE_DIR = UI_DIR / "state"
UI_LOG_DIR = UI_STATE_DIR / "logs"
UI_CONFIG_DIR = UI_STATE_DIR / "configs"
# Uploaded inputs (controlnet frames, ip-adapter refs, init frames) must live
# under data/ because the CLI resolves those paths relative to data/.
UI_INPUT_DIR = DATA_DIR / "ui_inputs"

OUTPUT_DIRS = {
    "Generate": REPO_ROOT / "output",
    "Upscale": REPO_ROOT / "upscaled",
    "Refine": REPO_ROOT / "refine",
    "Stylize": REPO_ROOT / "stylize",
}
STYLIZE_DIR = OUTPUT_DIRS["Stylize"]
WILDCARD_DIR = REPO_ROOT / "wildcards"
EMBEDDING_DIRS = [DATA_DIR / "embeddings", DATA_DIR / "sdxl_embeddings"]


def cli_python() -> Path:
    """Python interpreter of the main animatediff venv."""
    override = os.environ.get("ANIMATEDIFF_PYTHON")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return REPO_ROOT / "venv" / "Scripts" / "python.exe"
    return REPO_ROOT / "venv" / "bin" / "python"


def to_data_rel(path: Path | str) -> str:
    """Path as the CLI expects it in a config: relative to data/, forward slashes.

    Paths outside data/ are returned absolute (the CLI joins them onto data/,
    and joining an absolute path yields the absolute path).
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def from_data_rel(rel: str) -> Path:
    """Inverse of to_data_rel: resolve a config path against data/."""
    p = Path(rel)
    return p if p.is_absolute() else (DATA_DIR / p).resolve()


def ensure_dirs() -> None:
    for d in (UI_STATE_DIR, UI_LOG_DIR, UI_CONFIG_DIR, UI_INPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
