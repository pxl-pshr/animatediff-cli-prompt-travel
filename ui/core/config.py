"""Prompt-config schema knowledge: defaults, legacy repair, and friendly labels.

The CLI validates configs with a strict pydantic model (unknown keys are
rejected), and many map keys are read with [] (missing keys crash). So every
config the UI writes is a full deep-merge of DEFAULT_CONFIG and the user's
choices, and anything loaded from disk goes through normalize() first.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

# Top-level fields accepted by animatediff.settings.ModelConfig.
MODEL_CONFIG_FIELDS = [
    "name", "path", "vae_path", "motion_module", "context_schedule", "lcm_map",
    "gradual_latent_hires_fix_map", "compile", "tensor_interpolation_slerp", "seed", "scheduler",
    "scheduler_kwargs", "steps", "guidance_scale", "unet_batch_size", "clip_skip", "prompt_fixed_ratio", "head_prompt",
    "prompt_map", "tail_prompt", "n_prompt", "is_single_prompt_mode", "lora_map", "motion_lora_map",
    "ip_adapter_map", "img2img_map", "region_map", "controlnet_map", "upscale_config",
    "stylize_config", "output", "result",
]

# (label, value) — ordered by how often people want them.
SCHEDULERS = [
    ("DPM++ SDE Karras (recommended)", "k_dpmpp_sde"),
    ("DPM++ 2M Karras", "k_dpmpp_2m"),
    ("Euler a", "euler_a"),
    ("Euler", "euler"),
    ("DDIM", "ddim"),
    ("UniPC", "unipc"),
    ("DPM++ 2M SDE Karras", "k_dpmpp_2m_sde"),
    ("DPM++ 2M", "dpmpp_2m"),
    ("DPM++ SDE", "dpmpp_sde"),
    ("DPM++ 2M SDE", "dpmpp_2m_sde"),
    ("LCM (for LCM-LoRA / LCM motion modules)", "lcm"),
    ("Heun", "heun"),
    ("LMS", "lms"),
    ("LMS Karras", "k_lms"),
    ("DPM2", "dpm_2"),
    ("DPM2 Karras", "k_dpm_2"),
    ("DPM2 a", "dpm_2_a"),
    ("DPM2 a Karras", "k_dpm_2_a"),
    ("PNDM", "pndm"),
]
VALID_SCHEDULERS = {v for _, v in SCHEDULERS}
LEGACY_SCHEDULERS = {
    "k_dpmpp": "k_dpmpp_2m", "dpmpp": "dpmpp_2m", "k_euler": "euler", "k_euler_a": "euler_a",
    "euler_ancestral": "euler_a", "dpm++_2m_karras": "k_dpmpp_2m", "dpm++_sde_karras": "k_dpmpp_sde",
}

# ControlNet types the CLI knows (SD1.5), with friendly labels and what the
# preprocessor does so users know whether to upload raw footage or ready-made maps.
CONTROLNETS: list[tuple[str, str, str]] = [
    ("controlnet_openpose", "Pose (OpenPose)", "Follows body poses detected in your footage."),
    ("controlnet_depth", "Depth", "Keeps the 3D layout / distance of objects."),
    ("controlnet_softedge", "Soft edge", "Follows soft outlines; good general-purpose structure."),
    ("controlnet_lineart", "Line art", "Follows clean line drawings of the footage."),
    ("controlnet_lineart_anime", "Line art (anime)", "Line art tuned for anime-style footage."),
    ("controlnet_canny", "Canny edges", "Follows hard edges; very strict."),
    ("controlnet_scribble", "Scribble", "Follows rough sketches."),
    ("controlnet_normalbae", "Normal map", "Follows surface orientation."),
    ("controlnet_seg", "Segmentation", "Follows semantic regions (sky, person, wall...)."),
    ("controlnet_mlsd", "Straight lines (MLSD)", "Architecture / interiors."),
    ("controlnet_tile", "Tile", "Keeps colors and composition of the source; adds detail."),
    ("controlnet_ip2p", "Instruct-Pix2Pix", "Restyles the source while keeping it recognizable."),
    ("animatediff_controlnet", "AnimateDiff ControlNet", "Like ip2p, trained for video; great for replacing characters."),
    ("controlnet_shuffle", "Shuffle", "Transfers color/style from the images."),
    ("controlnet_inpaint", "Inpaint", "Fills masked areas."),
    ("qr_code_monster_v2", "QR Code Monster v2", "Hides a pattern/illusion inside the video."),
    ("qr_code_monster_v1", "QR Code Monster v1", "Older version of QR Code Monster."),
    ("controlnet_mediapipe_face", "MediaPipe face", "Follows facial landmarks."),
]
CONTROLNET_LABELS = {k: label for k, label, _ in CONTROLNETS}
CONTROLNET_KEYS = [k for k, _, _ in CONTROLNETS]

IP_ADAPTER_VARIANTS = [
    ("IP-Adapter Plus (default)", "plus"),
    ("IP-Adapter (standard)", "base"),
    ("IP-Adapter Light", "light"),
    ("IP-Adapter Plus Face", "plus_face"),
    ("IP-Adapter Full Face", "full_face"),
]

SPARSE_SCALE_LIST = [0.5, 0.4, 0.3, 0.2, 0.1]


def _cn_unit(enable: bool = False, scale: float = 1.0) -> dict:
    return {
        "enable": enable,
        "use_preprocessor": True,
        "guess_mode": False,
        "controlnet_conditioning_scale": scale,
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
        "control_scale_list": [],
        "control_region_list": [],
    }


DEFAULT_CONFIG: dict = {
    "name": "ui",
    "path": "",
    "vae_path": "",
    "motion_module": "",
    "context_schedule": "uniform",
    "lcm_map": {"enable": False, "start_scale": 0.15, "end_scale": 0.75,
                "gradient_start": 0.2, "gradient_end": 0.75},
    "gradual_latent_hires_fix_map": {"enable": False, "scale": {"0": 0.5, "0.7": 1.0},
                                     "reverse_steps": 5, "noise_add_count": 3},
    "compile": False,
    "tensor_interpolation_slerp": True,
    "seed": [-1],
    "scheduler": "k_dpmpp_sde",
    "scheduler_kwargs": {},
    "steps": 20,
    "guidance_scale": 8.0,
    "unet_batch_size": 1,
    "clip_skip": 2,
    "prompt_fixed_ratio": 0.5,
    "head_prompt": "",
    "prompt_map": {"0": ""},
    "tail_prompt": "",
    "n_prompt": ["(worst quality, low quality:1.4), text, watermark, signature, border"],
    "is_single_prompt_mode": False,
    "lora_map": {},
    "motion_lora_map": {},
    "ip_adapter_map": {
        "enable": False, "input_image_dir": "ip_adapter_image/test", "prompt_fixed_ratio": 0.5,
        "save_input_image": True, "resized_to_square": False, "scale": 0.5,
        "is_full_face": False, "is_plus_face": False, "is_plus": True, "is_light": False,
    },
    "img2img_map": {"enable": False, "init_img_dir": "init_imgs/sample0", "save_init_image": True,
                    "denoising_strength": 0.7},
    "region_map": {},
    "controlnet_map": {
        "input_image_dir": "controlnet_image/test",
        "max_samples_on_vram": 200,
        "max_models_on_vram": 3,
        "save_detectmap": True,
        "preprocess_on_gpu": True,
        "is_loop": True,
        **{k: _cn_unit() for k in CONTROLNET_KEYS},
        "controlnet_ref": {
            "enable": False, "ref_image": "ref_image/ref_sample.png",
            "attention_auto_machine_weight": 0.3, "gn_auto_machine_weight": 0.3, "style_fidelity": 0.5,
            "reference_attn": True, "reference_adain": False, "scale_pattern": [1.0],
        },
    },
    "upscale_config": {
        "scheduler": "k_dpmpp_sde", "steps": 20, "strength": 0.5, "guidance_scale": 10,
        "controlnet_tile": {"enable": True, "controlnet_conditioning_scale": 1.0, "guess_mode": False,
                            "control_guidance_start": 0.0, "control_guidance_end": 1.0},
        "controlnet_line_anime": {"enable": False, "controlnet_conditioning_scale": 1.0, "guess_mode": False,
                                  "control_guidance_start": 0.0, "control_guidance_end": 1.0},
        "controlnet_ip2p": {"enable": False, "controlnet_conditioning_scale": 0.5, "guess_mode": False,
                            "control_guidance_start": 0.0, "control_guidance_end": 1.0},
        "controlnet_ref": {"enable": False, "use_frame_as_ref_image": False, "use_1st_frame_as_ref_image": False,
                           "ref_image": "ref_image/ref_sample.png", "attention_auto_machine_weight": 1.0,
                           "gn_auto_machine_weight": 1.0, "style_fidelity": 0.25, "reference_attn": True,
                           "reference_adain": False},
    },
    "stylize_config": {},
    "output": {"format": "mp4", "fps": 8, "encode_param": {"crf": 10}},
    "result": {},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG)


def _slash(p) -> str:
    return str(p or "").replace("\\", "/")


def normalize(raw: dict) -> tuple[dict, list[str]]:
    """Repair a config loaded from disk so the CLI will accept it.

    Returns (config, notes) where notes describe every change made.
    """
    notes: list[str] = []
    raw = copy.deepcopy(raw or {})

    # Legacy animatediff-cli format: "prompt": [...] instead of prompt_map.
    if "prompt" in raw:
        legacy = raw.pop("prompt")
        if not raw.get("prompt_map"):
            first = legacy[0] if isinstance(legacy, list) and legacy else (legacy if isinstance(legacy, str) else "")
            raw["prompt_map"] = {"0": first}
            notes.append("Converted old-style 'prompt' list into the prompt timeline.")
    for key in list(raw):
        if key not in MODEL_CONFIG_FIELDS:
            raw.pop(key)
            notes.append(f"Removed unsupported field '{key}'.")

    sched = str(raw.get("scheduler", "") or "")
    if sched and sched not in VALID_SCHEDULERS:
        new = LEGACY_SCHEDULERS.get(sched, DEFAULT_CONFIG["scheduler"])
        notes.append(f"Scheduler '{sched}' is not valid; using '{new}'.")
        raw["scheduler"] = new

    if "seed" in raw and not isinstance(raw["seed"], list):
        raw["seed"] = [raw["seed"]]
    if isinstance(raw.get("n_prompt"), str):
        raw["n_prompt"] = [raw["n_prompt"]]
    if raw.get("n_prompt") == []:
        raw["n_prompt"] = [""]

    for key in ("path", "vae_path", "motion_module"):
        if key in raw:
            raw[key] = _slash(raw[key])
    for key in ("lora_map", "motion_lora_map"):
        if isinstance(raw.get(key), dict):
            raw[key] = {_slash(k): v for k, v in raw[key].items()}

    pm = raw.get("prompt_map")
    if isinstance(pm, dict):
        clean = {}
        for k, v in pm.items():
            try:
                clean[str(int(str(k).strip()))] = "" if v is None else str(v)
            except ValueError:
                notes.append(f"Dropped prompt keyframe with non-numeric frame '{k}'.")
        raw["prompt_map"] = dict(sorted(clean.items(), key=lambda kv: int(kv[0])))

    cfg = deep_merge(DEFAULT_CONFIG, raw)
    # region_map / stylize_config / result are free-form: keep exactly what was loaded.
    for free in ("region_map", "stylize_config", "result"):
        cfg[free] = copy.deepcopy(raw.get(free, DEFAULT_CONFIG[free]))
    return cfg, notes


def load_config_file(path: Path | str) -> tuple[dict, list[str]]:
    text = Path(path).read_text(encoding="utf-8")
    # Configs in the README carry '# comments'; strip them so they still load.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_strip_comments(text))
    return normalize(data)


def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        in_str = False
        esc = False
        for i, ch in enumerate(line):
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif ch == "#" and not in_str:
                line = line[:i]
                break
        out.append(line)
    text = "\n".join(out)
    return re.sub(r",(\s*[}\]])", r"\1", text)   # trailing commas


def dump(cfg: dict) -> str:
    return json.dumps(cfg, indent=2, ensure_ascii=False)


def slug(text: str, fallback: str = "ui") -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text or "").strip("_").lower()
    return s[:40] or fallback


def recommended_settings(mm) -> dict | None:
    """Settings that special motion modules need. `mm` is a library.ModelFile (or None)."""
    if mm is None or mm.kind != "motion_module":
        return None
    stem = mm.path.stem.lower()
    if "lightning" in stem:
        m = re.search(r"(\d+)\s*step", stem)
        n = int(m.group(1)) if m else 4
        return {
            "label": f"AnimateDiff-Lightning {n}-step",
            "scheduler": "euler", "steps": 3 if n == 2 else n, "guidance_scale": 1.0, "steps_tolerance": 1,
            "scheduler_kwargs": {"timestep_spacing": "trailing", "beta_schedule": "linear"},
            "note": "Distilled for speed: Euler, CFG 1 (negative prompt is ignored), "
                    f"{3 if n == 2 else n} steps. Camera motion LoRAs work at 0.7–0.8.",
        }
    if "animatelcm" in stem:
        return {
            "label": "AnimateLCM", "scheduler": "lcm", "steps": 6, "guidance_scale": 2.0, "steps_tolerance": 2,
            "note": "Use the LCM sampler, 4–8 steps, CFG 1–2, and add AnimateLCM_sd15_t2v_lora (weight 0.8) "
                    "under LoRAs. Leave Fast mode (LCM-LoRA) off.",
        }
    if mm.arch == "v3":
        return {
            "label": "AnimateDiff v3", "note": "Optional: add v3_sd15_adapter as a LoRA "
                                               "(0.5–1.0) — it can reduce flicker but adds a 'stock footage' look.",
        }
    return None


def ip_variant_flags(variant: str) -> dict:
    return {
        "is_plus": variant == "plus",
        "is_plus_face": variant == "plus_face",
        "is_full_face": variant == "full_face",
        "is_light": variant == "light",
    }


def ip_variant_of(ip_map: dict) -> str:
    if ip_map.get("is_plus_face"):
        return "plus_face"
    if ip_map.get("is_full_face"):
        return "full_face"
    if ip_map.get("is_light"):
        return "light"
    if ip_map.get("is_plus"):
        return "plus"
    return "base"
