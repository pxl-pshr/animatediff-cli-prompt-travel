"""Form values <-> CLI config, and staging of uploaded inputs into data/ui_inputs.

Form values are a plain dict (see FORM_DEFAULTS) so this module has no Gradio
dependency and can be tested on its own.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

from .config import (CONTROLNET_KEYS, SPARSE_SCALE_LIST, deep_merge, default_config, ip_variant_flags,
                     ip_variant_of, recommended_settings, slug)
from .library import LIBRARY
from .media import extract_frames, is_video, numbered_pngs, save_image_at_frame, spread_frames
from .paths import UI_INPUT_DIR, from_data_rel, to_data_rel

N_LORA = 4
N_MLORA = 2
N_CN = 3

ASSIGN_KEYFRAMES = "Match prompt keyframes, in order"
ASSIGN_SPREAD = "Spread evenly across the video"
ASSIGN_FIRST = "First image for the whole video"
ASSIGN_MODES = [ASSIGN_KEYFRAMES, ASSIGN_SPREAD, ASSIGN_FIRST]

SRC_VIDEO = "Video (every frame)"
SRC_IMAGES = "Images at keyframes"
SRC_FOLDER = "Existing folder"
SRC_MODES = [SRC_VIDEO, SRC_IMAGES, SRC_FOLDER]

NEIGHBOR_AUTO = "Auto"
NEIGHBOR_OFF = "Only its own frame"
NEIGHBOR_WIDE = "Spread to neighbors"
NEIGHBOR_MODES = [NEIGHBOR_AUTO, NEIGHBOR_OFF, NEIGHBOR_WIDE]


def _cn_defaults(i: int) -> dict:
    return {f"cn{i}_enable": False, f"cn{i}_type": "controlnet_openpose", f"cn{i}_source": SRC_VIDEO,
            f"cn{i}_video": None, f"cn{i}_images": None, f"cn{i}_folder": "", f"cn{i}_preprocess": True,
            f"cn{i}_weight": 1.0, f"cn{i}_start": 0.0, f"cn{i}_end": 1.0, f"cn{i}_neighbor": NEIGHBOR_AUTO}


FORM_DEFAULTS: dict = {
    "name": "ui", "ckpt": None, "mm": None, "vae": "",
    "head": "", "timeline": [[0, ""]], "tail": "", "neg": default_config()["n_prompt"][0],
    "width": 512, "height": 768, "length": 32, "fps": 8, "fmt": "mp4",
    "seed": -1, "steps": 20, "cfg_scale": 8.0, "scheduler": "k_dpmpp_sde", "clip_skip": 2, "repeats": 1,
    "context": 16, "overlap": 4, "stride": 0, "ctx_sched": "uniform", "fixed_ratio": 0.5,
    **{f"lora{i}": None for i in range(N_LORA)}, **{f"lora{i}_w": 0.8 for i in range(N_LORA)},
    **{f"mlora{i}": None for i in range(N_MLORA)}, **{f"mlora{i}_w": 1.0 for i in range(N_MLORA)},
    "lcm_enable": False, "lcm_start": 0.15, "lcm_end": 0.75, "lcm_gstart": 0.2, "lcm_gend": 0.75,
    "hires_enable": False, "hires_scale": 0.5, "hires_switch": 0.7, "hires_reverse": 5, "hires_noise": 3,
    "ip_enable": False, "ip_images": None, "ip_folder": "", "ip_assign": ASSIGN_KEYFRAMES,
    "ip_variant": "plus", "ip_scale": 0.5, "ip_square": False,
    **{k: v for i in range(N_CN) for k, v in _cn_defaults(i).items()},
    "cn_lowvram": False, "end_hold": False,
    "ref_enable": False, "ref_image": None, "ref_fidelity": 0.5,
    "i2i_enable": False, "i2i_video": None, "i2i_folder": "", "i2i_strength": 0.7,
}


def is_adapter(rel: str) -> bool:
    """AnimateDiff v3 Domain Adapter: an image LoRA, but in a format only the CLI's motion_lora_map loads."""
    m = LIBRARY.by_rel(rel or "")
    return bool(m and m.arch == "v3_adapter")


def split_loras(pairs) -> tuple[dict, dict]:
    """[(rel, weight)] from LoRA slots -> (lora_map, motion_lora_map entries for adapters)."""
    loras, adapters = {}, {}
    for rel, w in pairs:
        if rel:
            (adapters if is_adapter(rel) else loras)[rel] = round(float(w), 3)
    return loras, adapters


def controlnets_with_frames(cfg: dict) -> tuple[list[str], list[str]]:
    """Enabled ControlNet types split into (have frames, empty folder). The CLI silently skips
    empty ones, so many old configs 'enable' a ControlNet that never did anything."""
    cn = cfg.get("controlnet_map") or {}
    root = from_data_rel(cn["input_image_dir"]) if cn.get("input_image_dir") else None
    usable, empty = [], []
    for key in CONTROLNET_KEYS:
        if (cn.get(key) or {}).get("enable"):
            (usable if root and numbered_pngs(root / key) else empty).append(key)
    return usable, empty


def timeline_to_map(rows) -> dict[str, str]:
    out: dict[int, str] = {}
    for row in rows or []:
        if row is None or len(row) < 2:
            continue
        frame, prompt = row[0], row[1]
        if frame is None or (isinstance(frame, str) and not frame.strip()):
            if prompt and str(prompt).strip():
                frame = max(out) + 8 if out else 0
            else:
                continue
        try:
            f = max(0, int(float(frame)))
        except (TypeError, ValueError):
            continue
        out[f] = "" if prompt is None else str(prompt)
    if not out:
        out[0] = ""
    return {str(k): out[k] for k in sorted(out)}


def map_to_timeline(pm: dict) -> list[list]:
    rows = sorted(((int(k), v) for k, v in (pm or {}).items()), key=lambda r: r[0])
    return [[f, p] for f, p in rows] or [[0, ""]]


def _gallery_paths(value) -> list[str]:
    paths = []
    for item in value or []:
        if isinstance(item, (list, tuple)):
            item = item[0]
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        if item:
            paths.append(str(item))
    return paths


def form_to_config(base: dict, v: dict) -> dict:
    """Overlay form values on a base config (which keeps fields the form doesn't show)."""
    cfg = deep_merge(default_config(), base or {})
    cfg["name"] = slug(v["name"])
    cfg["path"] = v["ckpt"] or ""
    cfg["motion_module"] = v["mm"] or ""
    cfg["vae_path"] = v["vae"] or ""
    cfg["head_prompt"] = (v["head"] or "").strip()
    cfg["tail_prompt"] = (v["tail"] or "").strip()
    cfg["prompt_map"] = timeline_to_map(v["timeline"])
    if v.get("end_hold"):
        # The engine always morphs from the last keyframe back toward the first (it is built for loops).
        # Without a loop, hold the last keyframe's prompt until the final frame instead.
        pm = cfg["prompt_map"]
        last, end = max(int(k) for k in pm), int(v["length"]) - 1
        if last < end:
            pm[str(end)] = pm[str(last)]
    cfg["n_prompt"] = [(v["neg"] or "").strip()]
    cfg["scheduler"] = v["scheduler"]
    rec = recommended_settings(LIBRARY.by_rel(v["mm"] or ""))
    if rec and rec.get("scheduler_kwargs"):
        cfg["scheduler_kwargs"] = {**cfg.get("scheduler_kwargs", {}), **rec["scheduler_kwargs"]}
    cfg["steps"] = int(v["steps"])
    cfg["guidance_scale"] = float(v["cfg_scale"])
    cfg["clip_skip"] = int(v["clip_skip"])
    cfg["context_schedule"] = v["ctx_sched"]
    cfg["prompt_fixed_ratio"] = float(v["fixed_ratio"])

    # One seed per repeat: the CLI cycles through the seed list, so a single
    # -1 would give every repeat the same random seed.
    repeats = max(1, int(v["repeats"]))
    seed = int(v["seed"]) if v["seed"] not in (None, "") else -1
    cfg["seed"] = [seed] + [-1] * (repeats - 1)

    old_loras = (base or {}).get("lora_map") or {}
    cfg["lora_map"] = {}
    cfg["motion_lora_map"] = {}
    for i in range(N_LORA):
        rel = v.get(f"lora{i}")
        if not rel:
            continue
        w = round(float(v[f"lora{i}_w"]), 3)
        old = old_loras.get(rel)
        if is_adapter(rel):
            cfg["motion_lora_map"][rel] = w
        elif isinstance(old, dict) and isinstance(old.get("scale"), dict) and old["scale"]:
            # Region/scheduled LoRA: keep its regions and per-frame schedule, rescaled to the new weight.
            first = float(next(iter(old["scale"].values()))) or 1.0
            ratio = w / first
            cfg["lora_map"][rel] = {**old, "scale": {k: round(float(s) * ratio, 3) for k, s in old["scale"].items()}}
        else:
            cfg["lora_map"][rel] = w
    for i in range(N_MLORA):
        if v.get(f"mlora{i}"):
            cfg["motion_lora_map"][v[f"mlora{i}"]] = round(float(v[f"mlora{i}_w"]), 3)

    cfg["lcm_map"] = {"enable": bool(v["lcm_enable"]), "start_scale": float(v["lcm_start"]),
                      "end_scale": float(v["lcm_end"]), "gradient_start": float(v["lcm_gstart"]),
                      "gradient_end": float(v["lcm_gend"])}
    cfg["gradual_latent_hires_fix_map"] = {
        "enable": bool(v["hires_enable"]),
        "scale": {"0": float(v["hires_scale"]), str(round(float(v["hires_switch"]), 3)): 1.0},
        "reverse_steps": int(v["hires_reverse"]), "noise_add_count": int(v["hires_noise"]),
    }

    ip = cfg["ip_adapter_map"]
    ip.update({"enable": bool(v["ip_enable"]), "scale": float(v["ip_scale"]),
               "resized_to_square": bool(v["ip_square"]), **ip_variant_flags(v["ip_variant"])})

    cn = cfg["controlnet_map"]
    for key in CONTROLNET_KEYS:
        cn[key]["enable"] = False
    cn["is_loop"] = not v.get("end_hold")
    if v["cn_lowvram"]:
        cn["max_samples_on_vram"] = 0
        cn["max_models_on_vram"] = 0
    else:
        cn["max_samples_on_vram"] = 200
        cn["max_models_on_vram"] = 3
    for i in range(N_CN):
        if not v[f"cn{i}_enable"]:
            continue
        key = v[f"cn{i}_type"]
        unit = cn.setdefault(key, {})
        unit.update({
            "enable": True,
            "use_preprocessor": bool(v[f"cn{i}_preprocess"]),
            "controlnet_conditioning_scale": float(v[f"cn{i}_weight"]),
            "control_guidance_start": float(v[f"cn{i}_start"]),
            "control_guidance_end": float(v[f"cn{i}_end"]),
        })
        unit.setdefault("guess_mode", False)
        unit.setdefault("control_region_list", [])
        mode = v[f"cn{i}_neighbor"]
        dense = v[f"cn{i}_source"] == SRC_VIDEO
        if mode == NEIGHBOR_OFF or (mode == NEIGHBOR_AUTO and dense):
            unit["control_scale_list"] = []
        else:
            unit["control_scale_list"] = list(SPARSE_SCALE_LIST)
    ref = cn["controlnet_ref"]
    ref["enable"] = bool(v["ref_enable"])
    ref["style_fidelity"] = float(v["ref_fidelity"])
    if ref["enable"]:
        # Reference-only needs every condition layer (uncond, cond, one per region) in one UNet batch.
        regions = [k for k, r in (cfg.get("region_map") or {}).items()
                   if k != "background" and isinstance(r, dict) and r.get("enable")]
        layers = (2 if cfg["guidance_scale"] > 1 else 1) + len(regions)
        cfg["unet_batch_size"] = max(int(cfg.get("unet_batch_size", 1)), layers)

    cfg["img2img_map"].update({"enable": bool(v["i2i_enable"]), "denoising_strength": float(v["i2i_strength"])})
    cfg["output"] = {**cfg.get("output", {}), "format": v["fmt"], "fps": int(v["fps"])}
    cfg["output"].setdefault("encode_param", {"crf": 10})
    return cfg


def stage_inputs(cfg: dict, v: dict, *, width: int, height: int, length: int, fps: float,
                 tag: str) -> tuple[Path | None, list[str]]:
    """Copy/extract every uploaded input into data/ui_inputs/<stamp>/ and point cfg at it.

    Returns (staging dir or None, errors).
    """
    errors: list[str] = []
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(tag)}"
    root = UI_INPUT_DIR / stamp
    used = False
    keyframes = sorted(int(k) for k in cfg["prompt_map"])

    def assign(paths: list[str], mode: str) -> list[int]:
        if mode == ASSIGN_FIRST:
            return [0]
        if mode == ASSIGN_KEYFRAMES and keyframes:
            frames = keyframes[: len(paths)]
            if len(paths) > len(frames):   # more images than keyframes: spread the rest
                frames = sorted(set(frames) | set(spread_frames(len(paths), length)))[: len(paths)]
            return frames
        return spread_frames(len(paths), length)

    # ---- ControlNet units
    cn = cfg["controlnet_map"]
    cn_root = root / "controlnet_image"
    for i in range(N_CN):
        if not v[f"cn{i}_enable"]:
            continue
        key = v[f"cn{i}_type"]
        dest = cn_root / key
        src_mode = v[f"cn{i}_source"]
        try:
            if src_mode == SRC_VIDEO:
                vid = v[f"cn{i}_video"]
                if not vid:
                    errors.append(f"ControlNet unit {i + 1}: upload a video.")
                    continue
                extract_frames(vid, dest, fps=fps, width=width, height=height, max_frames=length)
            elif src_mode == SRC_IMAGES:
                imgs = _gallery_paths(v[f"cn{i}_images"])
                if not imgs:
                    errors.append(f"ControlNet unit {i + 1}: add at least one image.")
                    continue
                for path, frame in zip(imgs, assign(imgs, ASSIGN_KEYFRAMES)):
                    save_image_at_frame(path, dest, frame, (width, height))
            else:
                folder = from_data_rel(v[f"cn{i}_folder"]) if v[f"cn{i}_folder"] else None
                pngs = numbered_pngs(folder) if folder else []
                if not pngs:
                    errors.append(f"ControlNet unit {i + 1}: the folder has no numbered PNG frames.")
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                for p in pngs:
                    shutil.copy2(p, dest / p.name)
            used = True
        except Exception as e:
            errors.append(f"ControlNet unit {i + 1}: could not prepare input ({e}).")
    if any(v[f"cn{i}_enable"] for i in range(N_CN)):
        cn["input_image_dir"] = to_data_rel(cn_root)

    # ---- Reference-only
    if v["ref_enable"]:
        if v["ref_image"]:
            dest = save_image_at_frame(v["ref_image"], root / "ref_image", 0)
            cn["controlnet_ref"]["ref_image"] = to_data_rel(dest)
            used = True
        elif not from_data_rel(cn["controlnet_ref"].get("ref_image", "")).is_file():
            errors.append("Reference-only: add a reference image.")

    # ---- IP-Adapter
    ip = cfg["ip_adapter_map"]
    if v["ip_enable"]:
        imgs = _gallery_paths(v["ip_images"])
        if imgs:
            dest = root / "ip_adapter_image"
            for path, frame in zip(imgs, assign(imgs, v["ip_assign"])):
                save_image_at_frame(path, dest, frame)
            ip["input_image_dir"] = to_data_rel(dest)
            used = True
        elif v["ip_folder"]:
            ip["input_image_dir"] = v["ip_folder"]
        else:
            errors.append("Reference images (IP-Adapter): add at least one image.")

    # ---- img2img
    i2i = cfg["img2img_map"]
    if v["i2i_enable"]:
        if v["i2i_video"]:
            dest = root / "init_imgs"
            try:
                if is_video(v["i2i_video"]):
                    extract_frames(v["i2i_video"], dest, fps=fps, width=width, height=height, max_frames=length)
                else:
                    save_image_at_frame(v["i2i_video"], dest, 0, (width, height))
                i2i["init_img_dir"] = to_data_rel(dest)
                used = True
            except Exception as e:
                errors.append(f"Start-from-video: could not extract frames ({e}).")
        elif v["i2i_folder"]:
            i2i["init_img_dir"] = v["i2i_folder"]
        else:
            errors.append("Start-from-video (img2img): upload a video or pick a frames folder.")

    if not used and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    return (root if used else None), errors


def config_to_form(cfg: dict, meta: dict | None = None) -> dict:
    """Inverse of form_to_config, for loading presets / previous runs."""
    meta = meta or {}
    v = copy.deepcopy(FORM_DEFAULTS)
    v["name"] = cfg.get("name") or "ui"
    v["ckpt"] = cfg.get("path") or None
    v["mm"] = cfg.get("motion_module") or None
    v["vae"] = cfg.get("vae_path") or ""
    v["head"] = cfg.get("head_prompt", "")
    v["timeline"] = map_to_timeline(cfg.get("prompt_map"))
    v["tail"] = cfg.get("tail_prompt", "")
    v["neg"] = (cfg.get("n_prompt") or [""])[0]
    seeds = cfg.get("seed") or [-1]
    v["seed"] = int(seeds[0]) if seeds else -1
    v["steps"] = int(cfg.get("steps", 20))
    v["cfg_scale"] = float(cfg.get("guidance_scale", 8))
    v["scheduler"] = cfg.get("scheduler", "k_dpmpp_sde")
    v["clip_skip"] = int(cfg.get("clip_skip", 2))
    v["ctx_sched"] = cfg.get("context_schedule") or "uniform"
    v["fixed_ratio"] = float(cfg.get("prompt_fixed_ratio", 0.5))
    out = cfg.get("output") or {}
    v["fmt"] = out.get("format", "mp4") if out.get("format") in ("mp4", "gif", "webm") else "mp4"
    v["fps"] = int(out.get("fps", 8))

    motion = cfg.get("motion_lora_map") or {}
    loras = list((cfg.get("lora_map") or {}).items()) + [(r, w) for r, w in motion.items() if is_adapter(r)]
    for i, (rel, w) in enumerate(loras[:N_LORA]):
        v[f"lora{i}"] = rel
        if isinstance(w, dict):           # region/scheduled lora format: take the first scale
            sc = w.get("scale") or {}
            w = next(iter(sc.values()), 1.0) if isinstance(sc, dict) else sc
        v[f"lora{i}_w"] = float(w)
    cams = [(r, w) for r, w in motion.items() if not is_adapter(r)]
    for i, (rel, w) in enumerate(cams[:N_MLORA]):
        v[f"mlora{i}"], v[f"mlora{i}_w"] = rel, float(w)

    lcm = cfg.get("lcm_map") or {}
    v.update({"lcm_enable": bool(lcm.get("enable")), "lcm_start": lcm.get("start_scale", 0.15),
              "lcm_end": lcm.get("end_scale", 0.75), "lcm_gstart": lcm.get("gradient_start", 0.2),
              "lcm_gend": lcm.get("gradient_end", 0.75)})
    hr = cfg.get("gradual_latent_hires_fix_map") or {}
    sc = hr.get("scale") or {}
    hi = [float(k) for k, s in sc.items() if float(s) >= 1.0]
    v.update({"hires_enable": bool(hr.get("enable")),
              "hires_scale": float(sc.get("0", 0.5)) if sc else 0.5,
              "hires_switch": min(hi) if hi else 0.7,
              "hires_reverse": int(hr.get("reverse_steps", 5)), "hires_noise": int(hr.get("noise_add_count", 3))})

    ip = cfg.get("ip_adapter_map") or {}
    v.update({"ip_enable": bool(ip.get("enable")), "ip_folder": ip.get("input_image_dir", "") if ip.get("enable") else "",
              "ip_variant": ip_variant_of(ip), "ip_scale": float(ip.get("scale", 0.5)),
              "ip_square": bool(ip.get("resized_to_square"))})

    cn = cfg.get("controlnet_map") or {}
    enabled, _ = controlnets_with_frames(cfg)      # only units that actually have guide frames
    cn_root = cn.get("input_image_dir", "")
    for i, key in enumerate(enabled[:N_CN]):
        unit = cn[key]
        folder_rel = f"{cn_root.rstrip('/')}/{key}" if cn_root else ""
        v.update({f"cn{i}_enable": True, f"cn{i}_type": key, f"cn{i}_source": SRC_FOLDER,
                  f"cn{i}_folder": folder_rel, f"cn{i}_preprocess": bool(unit.get("use_preprocessor", True)),
                  f"cn{i}_weight": float(unit.get("controlnet_conditioning_scale", 1.0)),
                  f"cn{i}_start": float(unit.get("control_guidance_start", 0.0)),
                  f"cn{i}_end": float(unit.get("control_guidance_end", 1.0)),
                  f"cn{i}_neighbor": NEIGHBOR_WIDE if unit.get("control_scale_list") else NEIGHBOR_OFF})
    v["end_hold"] = False   # decided below from the timeline (is_loop alone is ambiguous in older configs)
    v["cn_lowvram"] = cn.get("max_models_on_vram", 3) == 0
    ref = cn.get("controlnet_ref") or {}
    v["ref_enable"] = bool(ref.get("enable"))
    v["ref_fidelity"] = float(ref.get("style_fidelity", 0.5))

    i2i = cfg.get("img2img_map") or {}
    v.update({"i2i_enable": bool(i2i.get("enable")), "i2i_strength": float(i2i.get("denoising_strength", 0.7)),
              "i2i_folder": i2i.get("init_img_dir", "") if i2i.get("enable") else ""})

    for key in ("width", "height", "length", "context", "overlap", "stride", "repeats"):
        if meta.get(key) is not None:
            v[key] = int(meta[key])
    rows = v["timeline"]
    if (meta.get("length") is not None and len(rows) >= 2 and rows[-1][0] == int(v["length"]) - 1
            and rows[-1][1] == rows[-2][1]):
        v["timeline"] = rows[:-1]          # the hold keyframe form_to_config adds for "end on the last keyframe"
        v["end_hold"] = True
    return v


def ui_meta_path(cfg_path: Path) -> Path:
    return cfg_path.with_name(cfg_path.stem + ".ui.json")


def write_ui_meta(cfg_path: Path, v: dict) -> None:
    meta = {k: v[k] for k in ("width", "height", "length", "context", "overlap", "stride", "repeats")}
    ui_meta_path(cfg_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_ui_meta(cfg_path: Path) -> dict:
    p = ui_meta_path(cfg_path)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
