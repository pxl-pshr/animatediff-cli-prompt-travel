"""Discover finished runs on disk (output/, upscaled/, refine/, stylize/)."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .media import numbered_pngs
from .paths import OUTPUT_DIRS, REPO_ROOT

VIDEO_SUFFIXES = {".mp4", ".webm", ".gif"}
FRAMES_DIR_RE = re.compile(r"^\d{2}-\d+(-upscaled)?$")   # 00-341774366206100, 00-123-upscaled
STAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")
# Folders inside a run/stylize project that never contain results.
NOISE_DIRS = {"00_controlnet_image", "01_controlnet_image", "00_img2img", "01_img2img", "00_ipadapter",
              "00_mask", "00_tmp_controlnet_image", "00_tmp_init_img_dir", "00_tmp_masked",
              "00_tmp_inpainted", "org_frame", "warp_img"}


@dataclass
class Run:
    path: Path
    category: str
    when: datetime
    videos: list[Path]
    frame_dirs: list[Path]
    config: dict

    @property
    def id(self) -> str:
        return str(self.path)

    @property
    def model(self) -> str:
        return Path(str(self.config.get("path", "")).replace("\\", "/")).stem

    @property
    def seed(self) -> str:
        if self.frame_dirs:
            return self.frame_dirs[0].name.split("-")[1]
        seeds = self.config.get("seed") or []
        return str(seeds[0]) if seeds else ""

    @property
    def frames(self) -> int:
        return len(numbered_pngs(self.frame_dirs[0])) if self.frame_dirs else 0

    @property
    def size(self) -> str:
        pngs = numbered_pngs(self.frame_dirs[0]) if self.frame_dirs else []
        if not pngs:
            return ""
        try:
            from PIL import Image
            with Image.open(pngs[0]) as im:
                return f"{im.width}x{im.height}"
        except Exception:
            return ""

    @property
    def preview(self) -> Path | None:
        """Best file to show: the main mp4/webm, else a gif, else None."""
        vids = sorted(self.videos, key=lambda p: ({".mp4": 0, ".webm": 1, ".gif": 2}[p.suffix.lower()],
                                                  len(p.name)))
        return vids[0] if vids else None

    @property
    def rel(self) -> str:
        try:
            return self.path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(self.path)

    def summary(self) -> str:
        c = self.config
        prompt = c.get("head_prompt") or ""
        pm = c.get("prompt_map") or {}
        if pm:
            first = pm.get("0") or next(iter(pm.values()), "")
            prompt = f"{prompt}, {first}" if prompt else first
        return prompt.strip(", ")[:160]


def _stamp(path: Path) -> datetime:
    m = STAMP_RE.search(path.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect(dir_: Path, category: str, depth: int, out: list[Run]) -> None:
    if depth < 0 or not dir_.is_dir() or dir_.name in NOISE_DIRS:
        return
    try:
        children = list(dir_.iterdir())
    except OSError:
        return
    videos = [p for p in children if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    frame_dirs = sorted(p for p in children if p.is_dir() and FRAMES_DIR_RE.match(p.name))
    if videos or frame_dirs:
        cfg_path = next((p for p in (dir_ / "prompt.json", dir_.parent / "prompt.json") if p.is_file()), None)
        out.append(Run(dir_, category, _stamp(dir_), videos, frame_dirs, _load_json(cfg_path) if cfg_path else {}))
    for c in children:
        if c.is_dir() and not FRAMES_DIR_RE.match(c.name):
            _collect(c, category, depth - 1, out)


def list_runs(categories: list[str] | None = None, limit: int = 300) -> list[Run]:
    runs: list[Run] = []
    for cat, root in OUTPUT_DIRS.items():
        if categories and cat not in categories:
            continue
        if root.is_dir():
            for d in root.iterdir():
                _collect(d, cat, 3, runs)
    runs.sort(key=lambda r: (r.when, r.path.name), reverse=True)
    return runs[:limit]


def find_run(path: str | Path) -> Run | None:
    path = Path(path)
    for r in list_runs():
        if r.path == path:
            return r
    return None


def category_of(path: Path) -> str:
    p = path.resolve()
    for cat, root in OUTPUT_DIRS.items():
        r = root.resolve()
        if p == r or r in p.parents:
            return cat
    return "Generate"


def run_from_dir(path: Path, category: str | None = None) -> Run | None:
    category = category or category_of(path)
    found: list[Run] = []
    _collect(path, category, 3, found)
    found.sort(key=lambda r: (r.when, r.path.name), reverse=True)
    return found[0] if found else None


def frame_dir_choices(limit: int = 80) -> list[tuple[str, str]]:
    """(label, path) for every frames folder of recent runs — sources for upscale/refine."""
    items = []
    for r in list_runs(limit=limit):
        for fd in r.frame_dirs:
            n = len(numbered_pngs(fd))
            label = f"{r.when:%m-%d %H:%M} · {r.category} · {r.model or r.path.name} · {r.size} · {n} frames"
            items.append((label, str(fd)))
    return items


def delete_run(run: Run) -> None:
    root_dirs = [d.resolve() for d in OUTPUT_DIRS.values()]
    target = run.path.resolve()
    if target in root_dirs or not any(root in target.parents for root in root_dirs):
        raise ValueError("Refusing to delete a folder outside the output directories.")
    shutil.rmtree(target)
