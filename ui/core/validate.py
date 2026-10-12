"""Pre-flight checks so mistakes surface in the UI instead of 2 minutes into a run."""
from __future__ import annotations

from pathlib import Path

from .config import CONTROLNET_LABELS, VALID_SCHEDULERS, recommended_settings
from .library import LIBRARY, ModelFile, pe_length
from .media import numbered_pngs
from .paths import from_data_rel


def max_context(mm: ModelFile | None) -> int:
    if mm is None:
        return 32
    if mm.arch == "v1":
        return 24
    if mm.arch == "v2_long":
        return pe_length(mm.path) or 32
    return 32


def check_generate(cfg: dict, *, width: int, height: int, length: int, context: int,
                   overlap: int | None = None, vram_gb: float = 0.0) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    ckpt = LIBRARY.by_rel(cfg.get("path", "")) if cfg.get("path") else None
    if not cfg.get("path"):
        errors.append("Choose a checkpoint (the image model).")
    elif not from_data_rel(cfg["path"]).is_file():
        errors.append(f"Checkpoint not found: data/{cfg['path']}")
    elif ckpt and ckpt.arch == "sd2":
        errors.append(f"{ckpt.path.name} is a Stable Diffusion 2.x model; AnimateDiff needs SD1.5 (or SDXL).")

    mm = LIBRARY.by_rel(cfg.get("motion_module", "")) if cfg.get("motion_module") else None
    if not cfg.get("motion_module"):
        errors.append("Choose a motion module.")
    elif not from_data_rel(cfg["motion_module"]).is_file():
        errors.append(f"Motion module not found: data/{cfg['motion_module']}")
    elif mm and mm.kind == "motion_lora":
        errors.append(f"{mm.path.name} is a motion LoRA, not a motion module. Add it under Motion LoRAs instead.")

    is_sdxl = bool(ckpt and ckpt.arch == "sdxl")
    if ckpt and mm:
        if is_sdxl and mm.arch != "sdxl_mm":
            errors.append("SDXL checkpoints need an SDXL motion module (e.g. mm_sdxl_v10_beta).")
        if not is_sdxl and mm.arch == "sdxl_mm":
            errors.append("This motion module is for SDXL; pick an SDXL checkpoint or an SD1.5 motion module.")

    if cfg.get("vae_path") and not from_data_rel(cfg["vae_path"]).is_file():
        errors.append(f"VAE not found: data/{cfg['vae_path']}")

    if cfg.get("scheduler") not in VALID_SCHEDULERS:
        errors.append(f"Unknown scheduler '{cfg.get('scheduler')}'.")

    # Prompts
    pm = cfg.get("prompt_map") or {}
    if not pm:
        errors.append("Add at least one prompt keyframe.")
    if not (cfg.get("head_prompt") or "").strip() and not any((v or "").strip() for v in pm.values()):
        errors.append("Write a prompt (head prompt or at least one keyframe).")
    late = sorted(int(k) for k in pm if int(k) >= length)
    if pm and len(late) == len(pm):
        errors.append(f"Every prompt keyframe is past the end of the video ({length} frames); add one at frame 0.")
    elif late:
        warns.append(f"Keyframes at frame {', '.join(map(str, late))} are past the end of the video "
                     f"({length} frames) and will never be reached.")
    if pm and "0" not in pm:
        warns.append("There is no keyframe at frame 0; the first keyframe's prompt will be used until then.")

    # Geometry
    if width % 8 or height % 8:
        errors.append("Width and height must be multiples of 8.")
    max_ctx = max_context(mm)
    if mm and context > max_ctx:
        warns.append(f"This motion module supports at most {max_ctx} context frames; it will be capped.")
    rec = recommended_settings(mm)
    if rec:
        if rec.get("guidance_scale") and cfg.get("guidance_scale", 0) > rec["guidance_scale"] + 1:
            warns.append(f"{rec['label']}: use CFG around {rec['guidance_scale']} (currently {cfg['guidance_scale']}).")
        if rec.get("steps") and abs(int(cfg.get("steps", 0)) - rec["steps"]) > rec.get("steps_tolerance", 2):
            warns.append(f"{rec['label']}: designed for about {rec['steps']} steps (currently {cfg['steps']}).")
    if context > length:
        warns.append(f"Context ({context}) is longer than the video ({length} frames); it will be reduced.")
    px = width * height
    if not is_sdxl and px > 768 * 768 and not cfg.get("gradual_latent_hires_fix_map", {}).get("enable"):
        warns.append("Large SD1.5 sizes often produce duplicated subjects. Consider generating smaller and "
                     "using Upscale / Refine, or enable the Gradual hires fix.")
    if vram_gb and vram_gb < 12 and px * context > 512 * 768 * 16:
        warns.append(f"This size/context may not fit in {vram_gb:.0f} GB VRAM; lower Context frames if it fails.")

    # LoRAs
    for rel in cfg.get("lora_map") or {}:
        lf = LIBRARY.by_rel(rel)
        if not from_data_rel(rel).is_file():
            errors.append(f"LoRA not found: data/{rel}")
        elif lf and lf.kind == "lycoris":
            warns.append(f"{lf.path.name} looks like a LyCORIS (LoHa/LoKr); these are not supported and may be ignored.")
        elif lf and ckpt and lf.arch and lf.arch != ckpt.arch and lf.kind == "lora":
            warns.append(f"LoRA {lf.path.name} is for {lf.arch.upper()} but the checkpoint is {ckpt.arch.upper()}.")
    for rel in cfg.get("motion_lora_map") or {}:
        ml = LIBRARY.by_rel(rel)
        if not from_data_rel(rel).is_file():
            errors.append(f"Motion LoRA not found: data/{rel}")
        elif ml and ml.arch == "v3_adapter":
            if mm and mm.arch != "v3":
                warns.append("The v3 Domain Adapter is meant for the v3 motion module.")
        elif mm and mm.arch == "v1":
            errors.append(f"Camera motion LoRA {Path(rel).name} only works with v2/v3 motion modules "
                          "(e.g. mm_sd_v15_v2). Remove it or switch motion module.")
        elif mm and mm.arch == "v3":
            warns.append(f"{Path(rel).name} was trained for v2; on v3 it applies only partially.")

    # LCM / hires / sampling
    lcm = cfg.get("lcm_map") or {}
    if lcm.get("enable"):
        if cfg.get("steps", 0) > 12:
            warns.append("LCM is enabled: 4–8 steps is enough; more steps mostly waste time.")
        if cfg.get("guidance_scale", 0) > 3:
            warns.append("LCM is enabled: use a low CFG (1.0–3.0) or results burn out.")
    hr = cfg.get("gradual_latent_hires_fix_map") or {}
    if hr.get("enable") and not lcm.get("enable") and cfg.get("scheduler") not in ("euler_a", "lcm"):
        warns.append("Gradual hires fix is designed for the 'Euler a' scheduler; other schedulers may look noisy.")
    if cfg.get("guidance_scale", 7) <= 1:
        warns.append("CFG ≤ 1 turns guidance off, so the negative prompt is ignored.")
    if overlap is not None and overlap >= context:
        errors.append("Overlap must be smaller than Context frames.")
    if cfg.get("context_schedule") not in ("uniform", "composite", "shuffle"):
        errors.append(f"Unknown context schedule '{cfg.get('context_schedule')}'.")
    dict_loras = [r for r, w in (cfg.get("lora_map") or {}).items() if isinstance(w, dict)]
    if dict_loras and (cfg.get("controlnet_map") or {}).get("controlnet_ref", {}).get("enable"):
        errors.append("Reference-only can't be combined with region/scheduled LoRAs.")

    # Inputs
    cn = cfg.get("controlnet_map") or {}
    cn_root = from_data_rel(cn.get("input_image_dir", "")) if cn.get("input_image_dir") else None
    for key, label in CONTROLNET_LABELS.items():
        unit = cn.get(key) or {}
        if unit.get("enable"):
            folder = cn_root / key if cn_root else None
            if not folder or not numbered_pngs(folder):
                errors.append(f"ControlNet '{label}' is enabled but has no input images.")
    ref = cn.get("controlnet_ref") or {}
    if ref.get("enable") and not from_data_rel(ref.get("ref_image", "")).is_file():
        errors.append("Reference-only is enabled but its reference image is missing.")

    ip = cfg.get("ip_adapter_map") or {}
    if ip.get("enable"):
        folder = from_data_rel(ip.get("input_image_dir", "")) if ip.get("input_image_dir") else None
        if not folder or not numbered_pngs(folder):
            errors.append("Reference images (IP-Adapter) are enabled but no images were added.")
        if is_sdxl and (ip.get("is_plus_face") or ip.get("is_full_face") or ip.get("is_light")):
            warns.append("For SDXL only the standard and Plus IP-Adapters exist.")

    i2i = cfg.get("img2img_map") or {}
    if i2i.get("enable"):
        folder = from_data_rel(i2i.get("init_img_dir", "")) if i2i.get("init_img_dir") else None
        n = len(numbered_pngs(folder)) if folder else 0
        if float(i2i.get("denoising_strength", 0.7)) * int(cfg.get("steps", 20)) < 1:
            errors.append("Start-from-video: denoising strength × steps is below 1, so nothing would be generated. "
                          "Raise the strength or the steps.")
        if not n:
            errors.append("Start-from-video (img2img) is enabled but there are no source frames.")
        elif n < length:
            warns.append(f"Only {n} source frames for a {length}-frame video; later frames start from noise.")

    if is_sdxl:
        warns.append("SDXL: ControlNet support is limited and Tile Upscale is not available.")
    return errors, warns
