"""Scan data/ for model files and classify them without loading any weights.

safetensors: the JSON header lists every tensor name.
.ckpt/.pt/.pth: these are zip archives whose data.pkl contains the state-dict
key names as plain strings, so a byte search is enough (nothing is unpickled).
"""
from __future__ import annotations

import collections
import io
import json
import os
import pickle
import struct
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_DIR, EMBEDDING_DIRS, WILDCARD_DIR, to_data_rel

MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}

# Folders the CLI manages itself (auto-downloaded helpers) or that hold inputs.
SKIP_DIRS = {
    "huggingface", "controlnet", "ip_adapter", "WD14tagger", "SAM", "DWPose",
    "softsplat", "anime_seg", "lcm_lora", "propainter", "ProPainter", "Grounding-DINO",
    "rife", "ui_inputs", "controlnet_image", "init_imgs", "ip_adapter_image", "mask",
    "ref_image", "sdxl_embeddings_cache", ".git",
}

V2_MM_KEY = "mid_block.motion_modules.0.temporal_transformer.norm.bias"


@dataclass(frozen=True)
class ModelFile:
    path: Path
    rel: str            # path relative to data/ as written into configs
    kind: str           # checkpoint | lora | lycoris | vae | motion_module | motion_lora | embedding | controlnet | unknown
    arch: str           # sd15 | sdxl | sd2 | v1 | v2 | sdxl_mm | ""
    size: int

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def label(self) -> str:
        tag = {"sdxl": " [SDXL]", "sd2": " [SD2 - unsupported]", "v1": " [v1]", "v2": " [v2]", "v3": " [v3]",
               "v2_long": " [v2 long]", "sdxl_mm": " [SDXL]", "v3_adapter": " [v3 domain adapter]"
               }.get(self.arch, "")
        return f"{self.path.name}{tag}"


def _safetensors_keys(path: Path) -> list[str]:
    return [k for k in _safetensors_header(path) if k != "__metadata__"]


class _TensorShape(tuple):
    pass


def _rebuild_tensor(storage, offset, size, *args, **kwargs):
    return _TensorShape(size)


class _ShapeUnpickler(pickle.Unpickler):
    """Reads tensor *shapes* from a torch data.pkl. Only whitelisted callables are
    resolved, so nothing from the file is ever executed."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor_v3", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a, **k: data
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

    def persistent_load(self, pid):
        return None


def _pickle_shapes(blob: bytes) -> dict | None:
    try:
        obj = _ShapeUnpickler(io.BytesIO(blob)).load()
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    return obj if isinstance(obj, dict) else None


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100_000_000:
            raise ValueError("bad safetensors header")
        return json.loads(f.read(n))


def pe_length(path: Path) -> int | None:
    """Stored positional-encoding length of a motion module (24 = v1, 32 = v2/v3, 64+ = long)."""
    try:
        if path.suffix == ".safetensors":
            header = _safetensors_header(path)
            key = next((k for k in header if k.endswith("pos_encoder.pe")), None)
            return int(header[key]["shape"][1]) if key else None
        blob = _pickle_blob(path)
        shapes = _pickle_shapes(blob) if blob else None
        if shapes:
            key = next((k for k in shapes if str(k).endswith("pos_encoder.pe")), None)
            if key and isinstance(shapes[key], tuple) and len(shapes[key]) >= 2:
                return int(shapes[key][1])
    except Exception:
        return None
    return None


def _pickle_blob(path: Path) -> bytes | None:
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("data.pkl")]
            if not names:
                return None
            return z.read(names[0])
    except (zipfile.BadZipFile, OSError, KeyError):
        return None


class _Probe:
    """Uniform 'does any key contain X' over either format."""

    def __init__(self, path: Path):
        self.keys: list[str] | None = None
        self.blob: bytes | None = None
        if path.suffix == ".safetensors":
            try:
                self.keys = _safetensors_keys(path)
            except Exception:
                self.keys = None
        else:
            self.blob = _pickle_blob(path)

    @property
    def readable(self) -> bool:
        return self.keys is not None or self.blob is not None

    def has(self, token: str) -> bool:
        if self.keys is not None:
            return any(token in k for k in self.keys)
        if self.blob is not None:
            return token.encode() in self.blob
        return False

    def startswith(self, prefix: str) -> bool:
        if self.keys is not None:
            return any(k.startswith(prefix) for k in self.keys)
        return self.has(prefix)


def classify(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    parent = path.parent.name.lower()
    p = _Probe(path)

    if p.readable:
        if p.has("emb_params") or p.has("string_to_param") or (p.has("clip_g") and size < 5_000_000):
            return "embedding", "sdxl" if p.has("clip_g") else "sd15"
        if p.has("motion_modules") or p.has("temporal_transformer"):
            if p.has("lora"):
                return "motion_lora", "v2"
            if "sdxl" in path.name.lower() or "hotshot" in path.name.lower():
                return "motion_module", "sdxl_mm"
            if p.has(V2_MM_KEY):
                pe = pe_length(path)
                return "motion_module", "v2_long" if pe and pe > 32 else "v2"
            pe = pe_length(path)
            if pe is None:
                return "motion_module", "v3" if "v3" in path.stem.lower() else "v1"
            return "motion_module", "v1" if pe <= 24 else "v3"
        if p.has("processor.to_q_lora"):
            # AnimateDiff v3 Domain Adapter: diffusers attn-processor LoRA, applied via motion_lora_map
            return "motion_lora", "v3_adapter"
        # Full checkpoints first: some merges carry leftover lora_* keys.
        if p.has("conditioner.embedders"):
            return "checkpoint", "sdxl"
        if p.has("model.diffusion_model"):
            return "checkpoint", "sd2" if p.has("cond_stage_model.model.") else "sd15"
        if p.has("control_model.") or p.has("controlnet_cond_embedding"):
            return "controlnet", ""
        if p.has("hada_w1") or p.has("lokr_w1"):
            return "lycoris", ""
        if p.startswith("lora_") or p.has("lora_down") or p.has("lora_up") or p.has(".lora_A") or p.has("lora.down"):
            is_xl = p.startswith("lora_te1_") or p.startswith("lora_te2_") or p.has("input_blocks")
            return "lora", "sdxl" if is_xl else "sd15"
        if p.has("encoder.down") and p.has("decoder.up") and not p.has("diffusion_model"):
            return "vae", ""

    # Unreadable (legacy pickle) or unrecognized: fall back on folder and size.
    if "motion" in parent:
        return ("motion_lora", "v2") if size < 300_000_000 else ("motion_module", "v1")
    if "vae" in parent:
        return "vae", ""
    if "lora" in parent or "lycoris" in parent:
        return "lora", "sd15"
    if "embedding" in parent:
        return "embedding", "sd15"
    if size > 1_500_000_000:
        return "checkpoint", "sd15"
    return "unknown", ""


class ModelLibrary:
    """Cached scan of data/. Re-scans only files whose size/mtime changed."""

    def __init__(self, root: Path = DATA_DIR):
        self.root = root
        self._cache: dict[str, tuple[float, int, str, str]] = {}
        self._files: list[ModelFile] = []
        self._lock = threading.Lock()

    def scan(self) -> list[ModelFile]:
        found: list[ModelFile] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=True):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                path = Path(dirpath) / fn
                if path.suffix.lower() not in MODEL_EXTS:
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                key = str(path)
                cached = self._cache.get(key)
                if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
                    kind, arch = cached[2], cached[3]
                else:
                    try:
                        kind, arch = classify(path)
                    except Exception:
                        kind, arch = "unknown", ""
                    self._cache[key] = (st.st_mtime, st.st_size, kind, arch)
                found.append(ModelFile(path, to_data_rel(path), kind, arch, st.st_size))
        found.sort(key=lambda m: m.path.name.lower())
        with self._lock:
            self._files = found
        return found

    @property
    def files(self) -> list[ModelFile]:
        with self._lock:
            files = self._files
        return files if files else self.scan()

    def of_kind(self, *kinds: str) -> list[ModelFile]:
        return [m for m in self.files if m.kind in kinds]

    def by_rel(self, rel: str) -> ModelFile | None:
        norm = rel.replace("\\", "/").strip().lower()
        for m in self.files:
            if m.rel.lower() == norm:
                return m
        return None

    def choices(self, *kinds: str) -> list[tuple[str, str]]:
        """(label, rel) pairs for gr.Dropdown."""
        return [(m.label, m.rel) for m in self.of_kind(*kinds)]


def list_embeddings() -> list[str]:
    names = []
    for d in EMBEDDING_DIRS:
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".pt", ".safetensors", ".bin"}:
                    names.append(p.stem)
    return names


def list_wildcards() -> list[str]:
    if not WILDCARD_DIR.is_dir():
        return []
    return sorted(f"__{p.stem}__" for p in WILDCARD_DIR.glob("*.txt"))


LIBRARY = ModelLibrary()
