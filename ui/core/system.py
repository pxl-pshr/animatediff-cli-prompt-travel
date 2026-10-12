"""Environment checks shown on the Setup tab and used to gate optional features."""
from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .media import which
from .paths import DATA_DIR, REPO_ROOT, cli_python

_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}

_PROBE = r"""
import json, importlib.util as u, sys
mods = ["torch", "diffusers", "transformers", "xformers", "controlnet_aux", "ffmpeg", "pandas",
        "onnxruntime", "segment_anything_hq", "groundingdino", "rembg", "cupy", "mediapipe"]
info = {m: bool(u.find_spec(m)) for m in mods}
try:
    import torch
    info["torch_version"] = torch.__version__
    info["cuda"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
except Exception as e:
    info["torch_error"] = repr(e)
try:
    import onnxruntime as ort
    info["ort_providers"] = ort.get_available_providers()
except Exception:
    info["ort_providers"] = []
try:
    from importlib.metadata import version
    info["diffusers_version"] = version("diffusers")
    info["animatediff_version"] = version("animatediff")
except Exception:
    pass
print("@@" + json.dumps(info))
"""


@lru_cache(maxsize=1)
def cli_env() -> dict:
    py = cli_python()
    if not py.exists():
        return {"ok": False, "error": f"Main venv not found at {py}"}
    try:
        out = subprocess.run([str(py), "-c", _PROBE], capture_output=True, text=True, timeout=180,
                             cwd=str(REPO_ROOT), **_NOWIN)
        line = next((l for l in out.stdout.splitlines() if l.startswith("@@")), None)
        if not line:
            return {"ok": False, "error": (out.stderr or out.stdout)[-500:]}
        info = json.loads(line[2:])
        info["ok"] = True
        return info
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def refresh_cli_env() -> dict:
    cli_env.cache_clear()
    return cli_env()


def gpu_info() -> list[dict]:
    smi = which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, **_NOWIN)
        gpus = []
        for line in out.stdout.strip().splitlines():
            name, total, used, driver = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "total_mb": int(total), "used_mb": int(used), "driver": driver})
        return gpus
    except Exception:
        return []


def vram_gb() -> float:
    g = gpu_info()
    return g[0]["total_mb"] / 1024 if g else 0.0


def rife_installed() -> bool:
    exe = DATA_DIR / "rife" / ("rife-ncnn-vulkan.exe" if sys.platform == "win32" else "rife-ncnn-vulkan")
    return exe.exists() and (DATA_DIR / "rife" / "rife-v4.6" / "flownet.bin").exists()


def features() -> dict[str, tuple[bool, str]]:
    """feature -> (available, how to enable)."""
    env = cli_env()
    has = lambda m: bool(env.get(m))
    pip = f'"{cli_python()}" -m pip install -e .'
    return {
        "Video → Video (stylize)": (has("pandas") and has("onnxruntime") and has("ffmpeg"),
                                    f"{pip}[stylize]"),
        "Fast pose / auto-tag on GPU (onnxruntime-gpu)": (
            "CUDAExecutionProvider" in env.get("ort_providers", []),
            f'Optional — they already work on CPU. "{cli_python()}" -m pip uninstall -y onnxruntime && '
            f'"{cli_python()}" -m pip install onnxruntime-gpu==1.16.3'),
        "Masks / regions / composite": (has("segment_anything_hq") and has("groundingdino"),
                                        f"{pip}[stylize_mask]"),
        "RIFE frame interpolation": (rife_installed(),
                                     "Download rife-ncnn-vulkan (Windows build) from github.com/nihui/rife-ncnn-vulkan/releases "
                                     "and extract it so data/rife/rife-ncnn-vulkan.exe and data/rife/rife-v4.6/ exist."),
        "ffmpeg": (bool(which("ffmpeg") and which("ffprobe")), "Install ffmpeg and add it to PATH."),
    }


def open_folder(path: Path | str) -> None:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        subprocess.Popen(["explorer", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
