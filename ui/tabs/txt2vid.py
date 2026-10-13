"""Text → Video (prompt travel) tab."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from components import help_label, issues_md, job_panel
from core import build as B
from core.config import (CONTROLNET_LABELS, CONTROLNETS, IP_ADAPTER_VARIANTS, SCHEDULERS, dump, load_config_file,
                         recommended_settings, slug)
from core.jobs import MANAGER, Job, cli_argv
from core.library import LIBRARY, list_embeddings, list_wildcards
from core.media import numbered_pngs
from core.outputs import list_runs, run_from_dir
from core.paths import OUTPUT_DIRS, PRESET_DIR, UI_CONFIG_DIR, UI_STATE_DIR
from core.system import open_folder, vram_gb
from core.validate import check_generate

USER_PRESET_DIR = PRESET_DIR / "ui_presets"
LAST_PATH = UI_STATE_DIR / "last_txt2vid.json"

SHAPES = {
    "Portrait 2:3 · 512×768": (512, 768),
    "Landscape 3:2 · 768×512": (768, 512),
    "Square · 512×512": (512, 512),
    "Tall 9:16 · 432×768": (432, 768),
    "Wide 16:9 · 768×432": (768, 432),
    "Wide 16:9 HD · 1024×576": (1024, 576),
    "Custom": None,
}
CTX_SCHEDULES = [("Uniform (default)", "uniform"), ("Composite (steadier, fewer layout changes)", "composite"),
                 ("Shuffle", "shuffle")]


# --------------------------------------------------------------------------- choices
def ckpt_choices():
    return LIBRARY.choices("checkpoint")


def mm_choices():
    return LIBRARY.choices("motion_module")


def vae_choices():
    return [("Use the checkpoint's own VAE", "")] + LIBRARY.choices("vae")


def lora_choices():
    # the v3 Domain Adapter is an image LoRA; build.py routes it to motion_lora_map for the CLI
    adapters = [(m.label, m.rel) for m in LIBRARY.of_kind("motion_lora") if m.arch == "v3_adapter"]
    return [("— none —", "")] + LIBRARY.choices("lora", "lycoris") + adapters


def mlora_choices():
    return [("— none —", "")] + [(m.label, m.rel) for m in LIBRARY.of_kind("motion_lora") if m.arch != "v3_adapter"]


def preset_choices():
    items = []
    if USER_PRESET_DIR.is_dir():
        for p in sorted(USER_PRESET_DIR.glob("*.json")):
            if not p.name.endswith(".ui.json"):
                items.append((f"★ My preset · {p.stem}", f"file:{p}"))
    for p in sorted(PRESET_DIR.glob("*.json")):
        if not p.name.endswith(".ui.json"):
            items.append((f"Example · {p.stem}", f"file:{p}"))
    for r in list_runs(["Generate"], limit=40):
        if (r.path / "prompt.json").is_file():
            items.append((f"Past run · {r.when:%Y-%m-%d %H:%M} · {r.model} · seed {r.seed}", f"run:{r.path}"))
    return items


def default_form() -> dict:
    v = dict(B.FORM_DEFAULTS)
    ck = ckpt_choices()
    if ck:
        v["ckpt"] = ck[0][1]
    mms = LIBRARY.of_kind("motion_module")
    pick = next((m for m in mms if "v15_v2" in m.name), None) or next((m for m in mms if m.arch == "v2"), None) \
        or (mms[0] if mms else None)
    v["mm"] = pick.rel if pick else None
    return v


def model_info(ckpt_rel: str, mm_rel: str) -> str:
    parts = []
    ck = LIBRARY.by_rel(ckpt_rel or "")
    mm = LIBRARY.by_rel(mm_rel or "")
    if ck:
        parts.append({"sdxl": "**SDXL** checkpoint", "sd2": "**SD 2.x** checkpoint — not supported"}
                     .get(ck.arch, "**SD 1.5** checkpoint"))
    elif ckpt_rel:
        parts.append("⚠️ checkpoint file not found")
    if mm:
        parts.append({
            "v1": "**v1** motion module — context up to 24, no camera LoRAs",
            "v2": "**v2** motion module — context up to 32, camera LoRAs work",
            "v2_long": "**long-context v2** motion module — context up to 64+",
            "v3": "**v3** motion module — context up to 32",
            "sdxl_mm": "**SDXL** motion module",
        }.get(mm.arch, "motion module"))
    elif mm_rel:
        parts.append("⚠️ motion module not found")
    text = " · ".join(parts)
    rec = recommended_settings(mm)
    if rec and rec.get("note"):
        text += f"\n\n💡 **{rec['label']}:** {rec['note']}"
    return text


def rec_button(mm_rel: str):
    rec = recommended_settings(LIBRARY.by_rel(mm_rel or ""))
    show = bool(rec and (rec.get("scheduler") or rec.get("steps")))
    return gr.update(visible=show, value=f"Apply {rec['label']} settings" if show else "")


def duration_text(length, fps) -> str:
    try:
        secs = int(length) / max(1, int(fps))
        return f"**{int(length)} frames** at {int(fps)} fps = **{secs:.1f} s** of video"
    except Exception:
        return ""


# --------------------------------------------------------------------------- loading
def load_source(source: str) -> tuple[dict, dict, list[str]]:
    """source is 'file:<path>' or 'run:<dir>'. Returns (config, ui meta, notes)."""
    kind, _, target = (source or "").partition(":")
    path = Path(target)
    if kind == "file":
        cfg, notes = load_config_file(path)
        return cfg, B.read_ui_meta(path), notes
    if kind == "run":
        cfg, notes = load_config_file(path / "prompt.json")
        meta = {}
        ui_meta = path / "ui_settings.json"
        if ui_meta.is_file():
            try:
                meta = json.loads(ui_meta.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        run = run_from_dir(path)
        if run and run.frame_dirs and "width" not in meta:
            pngs = numbered_pngs(run.frame_dirs[0])
            if pngs:
                from PIL import Image
                with Image.open(pngs[0]) as im:
                    meta.update({"width": im.width, "height": im.height, "length": len(pngs)})
        return cfg, meta, notes
    raise ValueError("Pick something to load first.")


# --------------------------------------------------------------------------- UI
def build(demo: gr.Blocks) -> SimpleNamespace:
    F: dict[str, gr.components.Component] = {}
    base_state = gr.State({})

    with gr.Row(equal_height=False):
        # ================================================================ left: settings
        with gr.Column(scale=3, min_width=520):
            # ---------------- presets: load and save together
            with gr.Group():
                with gr.Row():
                    preset = gr.Dropdown(preset_choices(), label="Preset", scale=6, value=None,
                                         info="Load an example, one of your presets, or the settings of a past run.",
                                         filterable=True)
                    load_btn = gr.Button("Load", scale=1, min_width=70)
                    presets_refresh = gr.Button("🔄", scale=0, min_width=44)
                with gr.Row():
                    save_name = gr.Textbox(show_label=False, placeholder="Preset name", max_lines=1,
                                           scale=6, container=False)
                    save_btn = gr.Button("Save", scale=1, min_width=70)
            load_notes = gr.Markdown(visible=False, elem_classes=["adui-warn"])

            # ---------------- model
            with gr.Group():
                with gr.Row():
                    F["ckpt"] = gr.Dropdown(ckpt_choices(), label="Checkpoint", scale=4,
                                            info="The image model: decides the look (photo, anime, painting...).",
                                            allow_custom_value=True, filterable=True)
                    F["mm"] = gr.Dropdown(mm_choices(), label="Motion module", scale=4,
                                          info="Decides how things move. v2/v3 are good defaults.",
                                          allow_custom_value=True, filterable=True)
                    refresh = gr.Button("🔄", scale=0, min_width=44)
                mm_info = gr.Markdown(elem_classes=["adui-info"])
                rec_btn = gr.Button("Apply recommended settings", size="sm", visible=False)

            # ---------------- prompt
            with gr.Group():
                F["head"] = gr.Textbox(label="Prompt", lines=2,
                                       placeholder="masterpiece, best quality, 1girl, silver hair, cyberpunk city at night",
                                       info="Used for every frame: subject, style, quality tags. Each frame's "
                                            "full prompt = this + its timeline row + the ending prompt."
                                            + (" Wildcards: " + ", ".join(list_wildcards()[:20]) if list_wildcards() else ""))
                help_label("Timeline", "Each row changes the prompt from that frame on, and the video morphs "
                                       "between rows. Example: 0 standing, 32 walking, 64 running.")
                F["timeline"] = gr.Dataframe(
                    value=[[0, ""]], headers=["Frame", "Prompt"], show_label=False,
                    datatype=["number", "str"], col_count=(2, "fixed"), row_count=(1, "dynamic"),
                    type="array", interactive=True, wrap=True, label="Timeline",
                    column_widths=["14%", "86%"])
                with gr.Row():
                    add_kf = gr.Button("＋ Row", size="sm")
                    spread_kf = gr.Button("↔ Spread evenly", size="sm")
                    sort_kf = gr.Button("Sort", size="sm")
                    clear_kf = gr.Button("Clear", size="sm")
                F["fixed_ratio"] = gr.Slider(0, 1, value=0.5, step=0.05, label="Keyframe hold",
                                             info="How long each row holds before morphing to the next "
                                                  "(0 = morph the whole time, 1 = near hard cut).")
                F["tail"] = gr.Textbox(label="Ending prompt", lines=1, max_lines=1,
                                       info="Optional. Added after every timeline row.")
                emb = list_embeddings()
                F["neg"] = gr.Textbox(label="Negative prompt", lines=2,
                                      info="What to avoid." + (" Embeddings you can use by name: " + ", ".join(emb)
                                                               if emb else ""))

            # ---------------- video
            with gr.Group():
                with gr.Row():
                    shape = gr.Dropdown(list(SHAPES), value="Portrait 2:3 · 512×768", label="Shape", scale=3)
                    F["width"] = gr.Slider(256, 1536, value=512, step=8, label="Width", scale=3)
                    F["height"] = gr.Slider(256, 1536, value=768, step=8, label="Height", scale=3)
                with gr.Row():
                    F["length"] = gr.Slider(8, 512, value=32, step=1, label="Frames", scale=3)
                    F["fps"] = gr.Slider(4, 30, value=8, step=1, label="FPS", scale=2)
                    F["fmt"] = gr.Radio(["mp4", "gif", "webm"], value="mp4", label="Format", scale=2)
                duration = gr.Markdown(duration_text(32, 8), elem_classes=["adui-info"])
                with gr.Row():
                    F["end_hold"] = gr.Checkbox(
                        label="Don't loop", value=False, scale=1,
                        info="Videos loop by default: the end blends back into the start. Tick to hold the "
                             "last timeline row to the end instead. For a clean loop: more Frames than Context "
                             "frames, evenly spaced rows, and no camera motion.")
                    F["hires_enable"] = gr.Checkbox(
                        label="Hi-res fix", value=False, scale=1,
                        info="For sizes above ~768 px: avoids duplicated subjects and renders faster.")

            # ---------------- quality & speed
            with gr.Group():
                with gr.Row():
                    F["steps"] = gr.Slider(1, 80, value=20, step=1, label="Steps", scale=2)
                    F["cfg_scale"] = gr.Slider(1, 20, value=8, step=0.5, label="CFG", scale=2,
                                                info="How strictly to follow the prompt. 7–10 is typical.")
                    F["scheduler"] = gr.Dropdown(SCHEDULERS, value="k_dpmpp_sde", label="Sampler", scale=3)
                with gr.Row():
                    F["lcm_enable"] = gr.Checkbox(label="Fast mode", value=False, scale=2,
                                                  info="LCM-LoRA: about 8 steps and low CFG. Great for quick drafts.")
                    F["seed"] = gr.Number(value=-1, precision=0, label="Seed", scale=2, minimum=-1,
                                          info="-1 = random. 🎲 resets to random, ♻️ reuses the last video's seed.")
                    rand_seed = gr.Button("🎲", scale=0, min_width=44)
                    reuse_seed = gr.Button("♻️", scale=0, min_width=44)

            # ---------------- creative add-ons
            with gr.Accordion("LoRAs & camera motion", open=False):
                for i in range(B.N_LORA):
                    with gr.Row():
                        F[f"lora{i}"] = gr.Dropdown(lora_choices(), value="", label=f"LoRA {i + 1}", scale=4,
                                                    allow_custom_value=True, filterable=True,
                                                    info=None if i else "Put the LoRA's trigger words in your prompt.")
                        F[f"lora{i}_w"] = gr.Slider(-1, 2, value=0.8, step=0.05, label="Weight", scale=2)
                for i in range(B.N_MLORA):
                    with gr.Row():
                        F[f"mlora{i}"] = gr.Dropdown(mlora_choices(), value="", label=f"Camera {i + 1}",
                                                     scale=4, allow_custom_value=True,
                                                     info=None if i else "Pan / zoom / roll (v2 & v3 motion modules). "
                                                                         "One-way moves can't loop seamlessly.")
                        F[f"mlora{i}_w"] = gr.Slider(0, 1.5, value=1.0, step=0.05, label="Strength", scale=2)

            with gr.Accordion("Match a look or character", open=False):
                F["ip_enable"] = gr.Checkbox(label="Reference images", value=False,
                                             info="IP-Adapter: steer the look or keep a character / face.")
                F["ip_images"] = gr.Gallery(label="Reference images", type="filepath", interactive=True,
                                            columns=5, height=180, file_types=["image"])
                F["ip_folder"] = gr.Textbox(visible=False)
                ip_folder_note = gr.Markdown(visible=False, elem_classes=["adui-info"])
                F["ip_assign"] = gr.Radio([("Timeline rows", B.ASSIGN_KEYFRAMES), ("Even spread", B.ASSIGN_SPREAD),
                                           ("First only", B.ASSIGN_FIRST)], value=B.ASSIGN_KEYFRAMES, label="Assign",
                                          info="Which frames use which image: one per timeline row (in order), "
                                               "spread evenly, or the first image for the whole video.")
                with gr.Row():
                    F["ip_variant"] = gr.Dropdown(IP_ADAPTER_VARIANTS, value="plus", label="Adapter type",
                                                  info="Plus = overall look. Face variants = keep a face/identity.")
                    F["ip_scale"] = gr.Slider(0, 1, value=0.5, step=0.05, label="Strength",
                                              info="0 = text only, 1 = image only. Keep the text prompt anyway.")
                    F["ip_square"] = gr.Checkbox(label="Square crop", value=False)
                F["ref_enable"] = gr.Checkbox(label="Reference-only", value=False,
                                              info="Copy the colors and style of one image.")
                with gr.Row():
                    F["ref_image"] = gr.Image(type="filepath", label="Reference image", height=180)
                    F["ref_fidelity"] = gr.Slider(0, 1, value=0.5, step=0.05, label="Fidelity")

            with gr.Accordion("ControlNet: follow a video, pose or edges", open=False):
                cn_desc = {k: d for k, _, d in CONTROLNETS}
                cn_parts = []
                with gr.Tabs():
                    for i in range(B.N_CN):
                        with gr.Tab(f"Unit {i + 1}"):
                            F[f"cn{i}_enable"] = gr.Checkbox(label="Enable", value=False)
                            with gr.Row():
                                F[f"cn{i}_type"] = gr.Dropdown([(l, k) for k, l, _ in CONTROLNETS],
                                                               value="controlnet_openpose", label="Type", scale=3)
                                F[f"cn{i}_source"] = gr.Radio([("Video", B.SRC_VIDEO), ("Images", B.SRC_IMAGES), ("Folder", B.SRC_FOLDER)],
                                                              value=B.SRC_VIDEO, label="Input", scale=4,
                                                              info="Video: every frame. Images: placed on the timeline "
                                                                   "rows in order. Folder: numbered PNGs under data/.")
                            desc = gr.Markdown(cn_desc["controlnet_openpose"], elem_classes=["adui-info"])
                            F[f"cn{i}_video"] = gr.Video(label="Video",
                                                         sources=["upload"], height=200)
                            F[f"cn{i}_images"] = gr.Gallery(label="Images",
                                                            type="filepath", interactive=True, columns=5, height=160,
                                                            file_types=["image"], visible=False)
                            F[f"cn{i}_folder"] = gr.Textbox(label="Folder", max_lines=1,
                                                            visible=False)
                            F[f"cn{i}_preprocess"] = gr.Checkbox(
                                value=True, label="Auto-detect",
                                info="Extract the pose / depth / edges from your footage. Untick if you upload "
                                     "ready-made maps.")
                            with gr.Row():
                                F[f"cn{i}_weight"] = gr.Slider(0, 2, value=1.0, step=0.05, label="Strength")
                                F[f"cn{i}_start"] = gr.Slider(0, 1, value=0.0, step=0.05, label="Start",
                                                               info="Fraction of the steps where it kicks in.")
                                F[f"cn{i}_end"] = gr.Slider(0, 1, value=1.0, step=0.05, label="End")
                            F[f"cn{i}_neighbor"] = gr.Radio(
                                [("Auto", B.NEIGHBOR_AUTO), ("Own frame", B.NEIGHBOR_OFF), ("Neighbors", B.NEIGHBOR_WIDE)],
                                value=B.NEIGHBOR_AUTO, label="Spread",
                                info="How far each guide image reaches. Auto: video frames only affect their own "
                                     "frame; sparse images also nudge nearby frames.")
                            cn_parts.append((F[f"cn{i}_type"], desc, F[f"cn{i}_source"], F[f"cn{i}_video"],
                                             F[f"cn{i}_images"], F[f"cn{i}_folder"]))
                F["cn_lowvram"] = gr.Checkbox(label="Low VRAM", value=False,
                                              info="Keep ControlNet models off the GPU. Slower; for 8–12 GB cards.")

            with gr.Accordion("Start from a video or image", open=False):
                F["i2i_enable"] = gr.Checkbox(label="Enable", value=False,
                                             info="Start from a source video or image instead of noise.")
                F["i2i_video"] = gr.File(label="Source", file_types=["video", "image"])
                F["i2i_folder"] = gr.Textbox(visible=False)
                i2i_folder_note = gr.Markdown(visible=False, elem_classes=["adui-info"])
                F["i2i_strength"] = gr.Slider(0.05, 1, value=0.7, step=0.05, label="Strength",
                                              info="How much to change it. Lower keeps more of the source.")

            # ---------------- advanced
            with gr.Accordion("Advanced", open=False):
                gr.Markdown("**Motion windows**", elem_classes=["adui-info"])
                with gr.Row():
                    F["context"] = gr.Slider(1, 32, value=16, step=1, label="Context frames",
                                             info="Frames the motion module sees at once. 16 is the sweet spot; "
                                                  "lower saves VRAM.")
                    F["overlap"] = gr.Slider(0, 12, value=4, step=1, label="Overlap",
                                             info="Frames shared between windows (usually context ÷ 4).")
                with gr.Row():
                    F["stride"] = gr.Slider(0, 4, value=0, step=1, label="Stride",
                                            info="Extra long-range windows. 0 is fine for most videos.")
                    F["ctx_sched"] = gr.Dropdown(CTX_SCHEDULES, value="uniform", label="Schedule")
                gr.Markdown("**Model**", elem_classes=["adui-info"])
                with gr.Row():
                    F["vae"] = gr.Dropdown(vae_choices(), value="", label="VAE", allow_custom_value=True, scale=3)
                    F["clip_skip"] = gr.Slider(1, 4, value=2, step=1, label="CLIP skip", scale=2)
                gr.Markdown("**Fast mode tuning**", elem_classes=["adui-info"])
                with gr.Row():
                    F["lcm_start"] = gr.Slider(0, 1, value=0.15, step=0.05, label="Start strength")
                    F["lcm_end"] = gr.Slider(0, 1, value=0.75, step=0.05, label="End strength")
                with gr.Row():
                    F["lcm_gstart"] = gr.Slider(0, 1, value=0.2, step=0.05, label="Ramp start",
                                                info="Fraction of the steps where the strength starts changing.")
                    F["lcm_gend"] = gr.Slider(0, 1, value=0.75, step=0.05, label="Ramp end")
                gr.Markdown("**Hi-res fix tuning**", elem_classes=["adui-info"])
                with gr.Row():
                    F["hires_scale"] = gr.Slider(0.25, 0.9, value=0.5, step=0.05, label="Draft scale",
                                                 info="Size of the early draft, as a fraction of the final size.")
                    F["hires_switch"] = gr.Slider(0.1, 0.95, value=0.7, step=0.05, label="Switch at",
                                                  info="Fraction of the steps where it switches to full size.")
                with gr.Row():
                    F["hires_reverse"] = gr.Slider(0, 10, value=5, step=1, label="Reverse steps")
                    F["hires_noise"] = gr.Slider(0, 10, value=3, step=1, label="Added noise")
                gr.Markdown("**Run & config**", elem_classes=["adui-info"])
                F["name"] = gr.Textbox(value="ui", label="Run name", max_lines=1, info="Used in the output folder name.")
                regions_md = gr.Markdown("No region prompts in this config.", elem_classes=["adui-info"])
                with gr.Row():
                    preview_btn = gr.Button("Show config", size="sm")
                    open_cfg = gr.Button("Open configs folder", size="sm")
                cfg_view = gr.Code(language="json", label="Config", interactive=False, visible=False, max_lines=40)
                import_file = gr.File(label="Import config (.json)", file_types=[".json"])

        # ================================================================ right: run + result
        with gr.Column(scale=2, min_width=380, elem_classes=["adui-sticky"]):
            gen_btn = gr.Button("Generate", variant="primary", elem_classes=["adui-generate"])
            F["repeats"] = gr.Slider(1, 16, value=1, step=1, label="Batch",
                                     info="How many videos to make. Each gets its own random seed "
                                          "(after the first, if you set one).")
            issues = gr.Markdown(visible=False)
            panel = job_panel("generate", "Video")
            with gr.Row(visible=False) as result_actions:
                upscale_this = gr.Button("Upscale / refine this", size="sm")

    form_inputs = set(F.values())

    # ------------------------------------------------------------------ helpers
    def values(data: dict) -> dict:
        return {name: data[comp] for name, comp in F.items()}

    def form_updates(v: dict) -> dict:
        out = {}
        for name, comp in F.items():
            if name in ("ip_images", "i2i_video", "ref_image") or name.endswith(("_video", "_images")):
                out[comp] = None          # uploads are not restored; folders are
            else:
                out[comp] = v.get(name, B.FORM_DEFAULTS.get(name))
        return out

    def extra_updates(v: dict, cfg: dict) -> dict:
        regions = [k for k in (cfg.get("region_map") or {}) if k != "background"]
        out = {
            mm_info: model_info(v.get("ckpt"), v.get("mm")),
            duration: duration_text(v["length"], v["fps"]),
            regions_md: (f"This config has **{len(regions)} region prompt(s)**. They are kept exactly as loaded "
                         "(edit them in the JSON file)." if regions else "No region prompts in this config."),
            ip_folder_note: gr.update(value=f"Using reference images from `data/{v['ip_folder']}` "
                                            "(upload new ones above to replace them).",
                                      visible=bool(v.get("ip_folder"))),
            i2i_folder_note: gr.update(value=f"Using source frames from `data/{v['i2i_folder']}`.",
                                       visible=bool(v.get("i2i_folder"))),
            shape: next((k for k, s in SHAPES.items() if s == (v["width"], v["height"])), "Custom"),
        }
        for i, (_, desc, _, vid, imgs, folder) in enumerate(cn_parts):
            src = v[f"cn{i}_source"]
            out[desc] = cn_desc.get(v[f"cn{i}_type"], "")
            out[vid] = gr.update(value=None, visible=src == B.SRC_VIDEO)
            out[imgs] = gr.update(value=None, visible=src == B.SRC_IMAGES)
            out[folder] = gr.update(value=v[f"cn{i}_folder"], visible=src == B.SRC_FOLDER)
        return out

    def apply_loaded(cfg: dict, meta: dict, notes: list[str]) -> dict:
        usable, empty = B.controlnets_with_frames(cfg)
        notes = list(notes)
        if empty:
            names = ", ".join(CONTROLNET_LABELS.get(k, k) for k in empty)
            notes.append(f"ControlNet {names}: switched on in this config but its folder has no images, so it "
                         "never had any effect. Left off.")
        if len(usable) > B.N_CN:
            notes.append(f"This config uses {len(usable)} ControlNets; only the first {B.N_CN} were loaded.")
        v = B.config_to_form(cfg, meta)
        out = form_updates(v)
        for i, (_, _, _, vid, imgs, folder) in enumerate(cn_parts):
            out.pop(vid, None)
            out.pop(imgs, None)
            out.pop(folder, None)
        out.update(extra_updates(v, cfg))
        out[base_state] = cfg
        out[load_notes] = gr.update(value="**Loaded with fixes:**\n" + "\n".join(f"- {n}" for n in notes),
                                    visible=bool(notes))
        out[issues] = gr.update(value="", visible=False)
        return out

    load_outputs = list(F.values()) + [base_state, load_notes, issues, mm_info, duration, regions_md, ip_folder_note,
                                       i2i_folder_note, shape] + [p[1] for p in cn_parts] + \
                   [p[3] for p in cn_parts] + [p[4] for p in cn_parts] + [p[5] for p in cn_parts]

    # ------------------------------------------------------------------ events: loading
    def on_load_preset(source):
        try:
            cfg, meta, notes = load_source(source)
        except Exception as e:
            raise gr.Error(f"Could not load: {e}")
        gr.Info("Settings loaded.")
        return apply_loaded(cfg, meta, notes)

    load_btn.click(on_load_preset, [preset], load_outputs)
    presets_refresh.click(lambda: gr.update(choices=preset_choices()), None, [preset])

    def on_import(file):
        if not file:
            return {}
        try:
            cfg, notes = load_config_file(file)
        except Exception as e:
            raise gr.Error(f"Not a valid config: {e}")
        return apply_loaded(cfg, {}, notes)

    import_file.upload(on_import, [import_file], load_outputs)

    def on_page_load():
        if LAST_PATH.is_file():
            try:
                cfg, notes = load_config_file(LAST_PATH)
                return apply_loaded(cfg, B.read_ui_meta(LAST_PATH), [])
            except Exception:
                pass
        v = default_form()
        out = form_updates(v)
        out.update(extra_updates(v, {}))
        out[base_state] = {}
        return out

    demo.load(on_page_load, None, load_outputs)

    # ------------------------------------------------------------------ events: small helpers
    def on_refresh():
        LIBRARY.scan()
        upd = [gr.update(choices=ckpt_choices()), gr.update(choices=mm_choices()), gr.update(choices=vae_choices())]
        upd += [gr.update(choices=lora_choices()) for _ in range(B.N_LORA)]
        upd += [gr.update(choices=mlora_choices()) for _ in range(B.N_MLORA)]
        gr.Info(f"Found {len(LIBRARY.of_kind('checkpoint'))} checkpoints, "
                f"{len(LIBRARY.of_kind('motion_module'))} motion modules, {len(LIBRARY.of_kind('lora'))} LoRAs.")
        return upd

    refresh.click(on_refresh, None, [F["ckpt"], F["mm"], F["vae"]] + [F[f"lora{i}"] for i in range(B.N_LORA)]
                  + [F[f"mlora{i}"] for i in range(B.N_MLORA)])
    for c in (F["ckpt"], F["mm"]):
        c.change(model_info, [F["ckpt"], F["mm"]], [mm_info], queue=False)
    F["mm"].change(rec_button, [F["mm"]], [rec_btn], queue=False)

    def apply_rec(mm_rel, sched, steps, cfg_scale, lcm):
        rec = recommended_settings(LIBRARY.by_rel(mm_rel or "")) or {}
        gr.Info(f"Applied {rec.get('label', 'recommended')} settings.")
        return (rec.get("scheduler", sched), rec.get("steps", steps), rec.get("guidance_scale", cfg_scale),
                False if rec.get("scheduler") else lcm)

    rec_btn.click(apply_rec, [F["mm"], F["scheduler"], F["steps"], F["cfg_scale"], F["lcm_enable"]],
                  [F["scheduler"], F["steps"], F["cfg_scale"], F["lcm_enable"]], queue=False)

    def on_shape(s, w, h):
        dims = SHAPES.get(s)
        return (dims[0], dims[1]) if dims else (w, h)

    shape.input(on_shape, [shape, F["width"], F["height"]], [F["width"], F["height"]], queue=False)
    for c in (F["width"], F["height"]):
        c.input(lambda w, h: next((k for k, s in SHAPES.items() if s == (w, h)), "Custom"),
                [F["width"], F["height"]], [shape], queue=False)
    for c in (F["length"], F["fps"]):
        c.change(duration_text, [F["length"], F["fps"]], [duration], queue=False)
    F["context"].input(lambda c: max(0, min(12, int(c) // 4)), [F["context"]], [F["overlap"]], queue=False)

    rand_seed.click(lambda: -1, None, [F["seed"]], queue=False)

    def reuse(run_path):
        """Seed of the video shown on the right, else of the newest generated video."""
        run = run_from_dir(Path(run_path)) if run_path else None
        if not run:
            runs = list_runs(["Generate"], limit=1)
            run = runs[0] if runs else None
        if run and run.seed.isdigit():
            gr.Info(f"Seed set to {run.seed} (from the {'shown' if run_path else 'latest'} video).")
            return int(run.seed)
        raise gr.Error("No previous video found.")

    reuse_seed.click(reuse, [panel.run_path], [F["seed"]])
    panel.run_path.change(lambda p: gr.update(visible=bool(p)), [panel.run_path], [result_actions], queue=False)

    def add_keyframe(rows, length):
        rows = [r for r in (rows or []) if r and r[0] not in (None, "")]
        frames = [int(float(r[0])) for r in rows] or [0]
        step = max(8, int(length) // 4)
        nxt = max(frames) + step if rows else 0
        return rows + [[min(nxt, max(0, int(length) - 1)), ""]]

    def spread(rows, length):
        rows = [r for r in (rows or []) if r and (r[0] not in (None, "") or (r[1] or "").strip())]
        n = len(rows) or 1
        return [[int(round(i * int(length) / n)), r[1]] for i, r in enumerate(rows)] or [[0, ""]]

    def sort_rows(rows):
        return B.map_to_timeline(B.timeline_to_map(rows))

    add_kf.click(add_keyframe, [F["timeline"], F["length"]], [F["timeline"]], queue=False)
    spread_kf.click(spread, [F["timeline"], F["length"]], [F["timeline"]], queue=False)
    sort_kf.click(sort_rows, [F["timeline"]], [F["timeline"]], queue=False)
    clear_kf.click(lambda: [[0, ""]], None, [F["timeline"]], queue=False)

    for i, (type_dd, desc, src, vid, imgs, folder) in enumerate(cn_parts):
        type_dd.change(lambda t: cn_desc.get(t, ""), [type_dd], [desc], queue=False)
        src.change(lambda s: (gr.update(visible=s == B.SRC_VIDEO), gr.update(visible=s == B.SRC_IMAGES),
                              gr.update(visible=s == B.SRC_FOLDER)), [src], [vid, imgs, folder], queue=False)
        # Uploading something implies the unit should be on.
        for comp in (vid, imgs):
            comp.upload(lambda: True, None, [F[f"cn{i}_enable"]], queue=False)
    F["ip_images"].upload(lambda: True, None, [F["ip_enable"]], queue=False)
    F["i2i_video"].upload(lambda: True, None, [F["i2i_enable"]], queue=False)
    F["ref_image"].upload(lambda: True, None, [F["ref_enable"]], queue=False)

    def on_lcm(enabled, steps, cfg_scale):
        if enabled:
            gr.Info("Fast mode: set 8 steps and CFG 2. Raise them a little if results look flat.")
            return 8, 2.0
        return (20 if steps <= 10 else steps), (8.0 if cfg_scale <= 3 else cfg_scale)

    F["lcm_enable"].input(on_lcm, [F["lcm_enable"], F["steps"], F["cfg_scale"]], [F["steps"], F["cfg_scale"]])

    def on_hires(enabled, sched, lcm):
        if enabled and not lcm and sched != "euler_a":
            gr.Info("Gradual hires fix works best with 'Euler a' — switched the sampler.")
            return "euler_a"
        return sched

    F["hires_enable"].input(on_hires, [F["hires_enable"], F["scheduler"], F["lcm_enable"]], [F["scheduler"]])

    # ------------------------------------------------------------------ events: config preview / presets
    def current_config(data) -> tuple[dict, dict]:
        v = values(data)
        return B.form_to_config(data[base_state], v), v

    def on_preview(data):
        cfg, _ = current_config(data)
        return gr.update(value=dump(cfg), visible=True)

    preview_btn.click(on_preview, form_inputs | {base_state}, [cfg_view])
    open_cfg.click(lambda: open_folder(UI_CONFIG_DIR), None, None, queue=False)

    def on_save(data):
        name = slug(data[save_name], "")
        if not name:
            raise gr.Error("Type a name for the preset.")
        cfg, v = current_config(data)
        USER_PRESET_DIR.mkdir(parents=True, exist_ok=True)
        path = USER_PRESET_DIR / f"{name}.json"
        path.write_text(dump(cfg), encoding="utf-8")
        B.write_ui_meta(path, v)
        gr.Info(f"Saved preset '{name}'. Uploaded images/videos are not part of presets.")
        return gr.update(choices=preset_choices(), value=f"file:{path}")

    save_btn.click(on_save, form_inputs | {base_state, save_name}, [preset])

    # ------------------------------------------------------------------ events: generate
    def on_generate(data):
        v = values(data)
        width, height = int(v["width"]) // 8 * 8, int(v["height"]) // 8 * 8
        length = int(v["length"])
        context = max(1, min(int(v["context"]), length))
        overlap = max(0, min(int(v["overlap"]), context - 1))
        stride = int(v["stride"])
        repeats = int(v["repeats"])
        v.update(width=width, height=height, length=length, context=context, overlap=overlap)

        cfg = B.form_to_config(data[base_state], v)
        staged, stage_errors = B.stage_inputs(cfg, v, width=width, height=height, length=length,
                                              fps=int(v["fps"]), tag=cfg["name"])
        errors, warns = check_generate(cfg, width=width, height=height, length=length, context=context,
                                       overlap=overlap, vram_gb=vram_gb())
        # staging already explains missing uploads; drop the validator's duplicate for the same feature
        topics = [t for t in ("Reference images (IP-Adapter)", "ControlNet", "Start-from-video", "Reference-only")
                  if any(e.startswith(t) for e in stage_errors)]
        errors = stage_errors + [e for e in errors if not any(e.startswith(t) for t in topics)]
        if errors:
            if staged:
                shutil.rmtree(staged, ignore_errors=True)
            return {issues: gr.update(value=issues_md(errors, warns), visible=True)}

        stamp = time.strftime("%Y%m%d-%H%M%S")
        UI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path = UI_CONFIG_DIR / f"{stamp}-{cfg['name']}.json"
        cfg_path.write_text(dump(cfg), encoding="utf-8")
        B.write_ui_meta(cfg_path, v)
        LAST_PATH.write_text(dump(cfg), encoding="utf-8")
        B.write_ui_meta(LAST_PATH, v)

        meta = {k: v[k] for k in ("width", "height", "length", "context", "overlap", "stride", "repeats")}

        def after(job: Job):
            if job.save_dir and job.save_dir.is_dir():
                (job.save_dir / "ui_settings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        ck = Path(cfg["path"]).stem
        job = MANAGER.submit(Job(
            kind="generate",
            title=f"Text → Video · {ck} · {width}×{height} · {length} frames" + (f" ×{repeats}" if repeats > 1 else ""),
            argv=cli_argv("generate", "-c", cfg_path, "-W", width, "-H", height, "-L", length, "-C", context,
                          "-O", overlap, "-S", stride, "-r", repeats, "-o", OUTPUT_DIRS["Generate"]),
            out_root=OUTPUT_DIRS["Generate"], on_success=after, meta={"config": str(cfg_path)},
        ))
        if MANAGER.queue_position(job) > 1 or (MANAGER.current() and MANAGER.current() is not job):
            gr.Info("Added to the queue — it starts when the current job finishes.")
        return {
            issues: gr.update(value=issues_md([], warns), visible=bool(warns)),
            panel.job_id: job.id,
            panel.sig: None,
        }

    gen_btn.click(on_generate, form_inputs | {base_state}, [issues, panel.job_id, panel.sig])

    def load_into_form(source: str) -> dict:
        out = on_load_preset(source)
        out[preset] = gr.update(choices=preset_choices(), value=source)
        return out

    return SimpleNamespace(panel=panel, upscale_this=upscale_this, F=F, preset=preset,
                           load_outputs=load_outputs + [preset], load_into_form=load_into_form)
