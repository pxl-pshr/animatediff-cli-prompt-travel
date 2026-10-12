"""ffmpeg / image helpers for turning uploads into the folder layouts the CLI expects.

The CLI reads frame-indexed PNGs: a file named 00000016.png applies to frame 16.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float

    @property
    def frames(self) -> int:
        return int(self.duration * self.fps)


def probe_video(path: Path | str) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True, **_NOWIN,
    )
    data = json.loads(out.stdout)
    s = data["streams"][0]

    def rate(v: str) -> float:
        try:
            n, d = v.split("/")
            return float(n) / float(d) if float(d) else 0.0
        except Exception:
            return 0.0

    fps = rate(s.get("avg_frame_rate", "0/0")) or rate(s.get("r_frame_rate", "0/0")) or 30.0
    dur = float(s.get("duration") or data.get("format", {}).get("duration") or 0)
    return VideoInfo(int(s["width"]), int(s["height"]), fps, dur)


def extract_frames(video: Path | str, out_dir: Path, *, fps: float, width: int, height: int,
                   start: float = 0.0, max_frames: int | None = None, start_index: int = 0) -> int:
    """Extract frames at `fps`, cover-scaled and center-cropped to width x height.

    Files are written as 00000000.png, 00000001.png, ... (offset by start_index).
    Returns the number of frames written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    vf = (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
          f"crop={width}:{height}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video), "-vf", vf]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [str(tmp / "%08d.png")]
    proc = subprocess.run(cmd, capture_output=True, text=True, **_NOWIN)
    if proc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-400:]}")
    files = sorted(tmp.glob("*.png"))
    for i, f in enumerate(files):
        f.replace(out_dir / f"{start_index + i:08d}.png")
    shutil.rmtree(tmp, ignore_errors=True)
    return len(files)


def save_image_at_frame(src: Path | str | Image.Image, out_dir: Path, frame: int,
                        size: tuple[int, int] | None = None) -> Path:
    """Save an image as <frame>.png, optionally cover-fit to size (w, h)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = src if isinstance(src, Image.Image) else Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if size:
        img = ImageOps.fit(img, size, Image.LANCZOS)
    dest = out_dir / f"{frame:08d}.png"
    img.save(dest)
    return dest


def numbered_pngs(folder: Path) -> list[Path]:
    if not folder or not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.png") if p.stem.isdigit())


def frame_numbers(folder: Path) -> list[int]:
    return [int(p.stem) for p in numbered_pngs(folder)]


def is_video(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_image(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def spread_frames(count: int, length: int) -> list[int]:
    """Evenly spaced frame indices for `count` images across `length` frames."""
    if count <= 0:
        return []
    if count == 1:
        return [0]
    step = (length - 1) / (count - 1) if length > 1 else 0
    return sorted({int(round(i * step)) for i in range(count)})


def frames_to_video(frames_dir: Path, out: Path, fps: float, crf: int = 18) -> Path:
    """Encode numbered PNGs to mp4 (used for previews of image-sequence results)."""
    pngs = numbered_pngs(frames_dir)
    if not pngs:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    lst = frames_dir / "_concat.txt"
    lst.write_text("".join(f"file '{p.name}'\nduration {1 / fps:.6f}\n" for p in pngs), encoding="utf-8")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
           "-i", str(lst), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-r", str(fps),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, **_NOWIN)
    lst.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-400:]}")
    return out
