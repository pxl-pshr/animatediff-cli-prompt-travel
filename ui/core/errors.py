"""Turn CLI tracebacks into a one-line cause plus an actionable hint."""
from __future__ import annotations

import re

EXC_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|NotFound\w*)):?\s*(.*)$")

# (pattern searched in the whole tail, short cause, hint)
KNOWN = [
    (r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED",
     "Ran out of GPU memory",
     "Lower the resolution or the Context frames (e.g. 16 → 12 or 8), turn off extra ControlNets, "
     "or enable 'Low VRAM' in the ControlNet panel. Generate small, then use Upscale / Refine."),
    (r"value is not a valid enumeration member.*permitted: 'lcm', 'ddim'|Invalid scheduler",
     "Unknown scheduler name in the config",
     "Pick a scheduler from the dropdown. Old configs sometimes say 'k_dpmpp', which is not valid "
     "(use 'k_dpmpp_2m' or 'k_dpmpp_sde')."),
    (r"validation error for ModelConfig",
     "The config file has an invalid value",
     "Load the config in the UI and re-save it; the UI fills in required fields and fixes types."),
    (r"no attribute 'to_to_k'",
     "This checkpoint needs a one-time fix",
     "Go to Tools → Fix checkpoint and run it on this model, then try again."),
    (r"model_index\.json|stable-diffusion-v1-5.*(not found|No such file)",
     "The SD1.5 base model download is missing or incomplete",
     "Check your internet connection; the base model downloads from Hugging Face on first run "
     "into data/models/huggingface."),
    (r"(FileNotFoundError|No such file or directory).*(\.safetensors|\.ckpt|\.pth|\.pt)\b",
     "A model file in the config does not exist",
     "Re-select the checkpoint / motion module / LoRA in the UI so the path points at a real file."),
    (r"RIFE model dir .* does not have a model|rife-ncnn-vulkan",
     "Frame interpolation (RIFE) is not installed",
     "Download rife-ncnn-vulkan for Windows and extract it so data/rife/rife-ncnn-vulkan.exe exists "
     "(with the rife-v4.6 model folder next to it), or set the interpolation multiplier to 1."),
    (r"ffmpeg.*(not found|is not recognized)|No such file or directory: 'ffmpeg'",
     "ffmpeg is not installed or not on PATH",
     "Install ffmpeg and make sure `ffmpeg -version` works in a new terminal."),
    (r"No module named '(segment_anything_hq|groundingdino|rembg|pandas|onnxruntime|cupy)'",
     "An optional feature's dependencies are not installed",
     "See Setup → Optional features for the one-line install command."),
    (r"list index out of range.*\n.*glob|IndexError: list index out of range",
     "Expected frames were not found",
     "A frames folder is empty. Check that the source folder contains numbered PNGs (00000000.png, ...)."),
    (r"not an original-format AnimateDiff motion module",
     "This motion module file is in diffusers format",
     "Download the original / ComfyUI release of the module instead (e.g. the *_comfyui.safetensors "
     "files for AnimateDiff-Lightning)."),
    (r"size mismatch|is missing \d+ weights for this architecture",
     "The motion module does not match the expected architecture",
     "Check that the motion module matches the checkpoint (SD1.5 vs SDXL). v1, v2, v3, AnimateLCM, "
     "Lightning and LongAnimateDiff SD1.5 modules are supported."),
    (r"Expected all tensors to be on the same device|CUDA error: device-side assert",
     "GPU error inside the pipeline",
     "Try again; if it repeats, restart the UI so the GPU is fully released."),
]


def explain_failure(lines: list[str], returncode: int | None) -> tuple[str, str]:
    tail = "\n".join(lines[-120:])
    for pattern, cause, hint in KNOWN:
        if re.search(pattern, tail, flags=re.IGNORECASE):
            detail = _last_exception(lines)
            return (f"{cause}" + (f" ({detail})" if detail and len(detail) < 160 else ""), hint)
    detail = _last_exception(lines)
    if detail:
        return detail[:300], "Open the log below for the full traceback."
    if returncode is not None and (returncode < 0 or returncode in (3221225477, -1073741819)):
        return "The process crashed (access violation)", "Usually a driver/VRAM problem; try a smaller resolution."
    return f"The CLI exited with code {returncode}", "Open the log below for details."


def _last_exception(lines: list[str]) -> str:
    for i in range(len(lines) - 1, max(-1, len(lines) - 80), -1):
        m = EXC_RE.match(lines[i].strip())
        if m:
            msg = m.group(2).strip()
            # pydantic puts the useful part on the next lines
            if "validation error" in msg and i + 2 < len(lines):
                msg += " — " + " ".join(l.strip() for l in lines[i + 1:i + 4])
            name = m.group(1).split(".")[-1]
            return f"{name}: {msg}" if msg else name
    return ""
