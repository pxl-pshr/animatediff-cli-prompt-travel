"""Upscale & Refine tab: `animatediff refine` and `animatediff tile-upscale` on finished frames."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
from PIL import Image

from components import issues_md, job_panel
from core.config import dump, load_config_file
from core.jobs import MANAGER, Job, cli_argv
from core.library import LIBRARY
from core.media import numbered_pngs
from core.outputs import frame_dir_choices, run_from_dir
from core.paths import OUTPUT_DIRS, UI_CONFIG_DIR
from core.system import features

SAME = "same-as-source"   # dropdowns can't display a choice whose value is empty
REFINE = "Refine"
TILE = "Tile upscale"


def source_for_run(run_path) -> str | None:
    if not run_path:
        return None
    run = run_from_dir(Path(run_path))
    return str(run.frame_dirs[0]) if run and run.frame_dirs else None


def _src_info(src: str | None):
    """(preview video, info markdown, default target width) for a frames folder."""
    if not src or not Path(src).is_dir():
        return None, "Pick the frames of a finished video.", 768
    pngs = numbered_pngs(Path(src))
    if not pngs:
        return None, "⚠️ This folder has no numbered PNG frames.", 768
    with Image.open(pngs[0]) as im:
        w, h = im.size
    run = run_from_dir(Path(src).parent)
    preview = str(run.preview) if run and run.preview and run.preview.suffix != ".gif" else None
    cfg_ok = (Path(src).parent / "prompt.json").is_file()
    info = f"**{len(pngs)} frames · {w}×{h}**" + ("" if cfg_ok else
                                                   " · ⚠️ no prompt.json next to these frames — pick a config below")
    return preview, info, min(2048, int(w * 1.5) // 8 * 8)


def select_source(src: str):
    preview, info, width = _src_info(src)
    return (gr.update(choices=frame_dir_choices(), value=src), gr.update(value=preview, visible=bool(preview)),
            info, width)


def build(demo: gr.Blocks) -> SimpleNamespace:
    rife_ok = features()["RIFE frame interpolation"][0]

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=480):
            with gr.Group():
                with gr.Row():
                    source = gr.Dropdown(frame_dir_choices(), label="Source", scale=6, info="A finished video (its frames folder).",
                                         allow_custom_value=True, filterable=True)
                    refresh = gr.Button("🔄", scale=0, min_width=44)
                src_info = gr.Markdown("Pick the frames of a finished video.", elem_classes=["adui-info"])
                src_preview = gr.Video(label="Source", visible=False, interactive=False, height=260, loop=True)

            method = gr.Radio([REFINE, TILE], value=REFINE, label="Method",
                          info="Refine: re-render larger with AnimateDiff (smooth, best detail). "
                               "Tile upscale: frame by frame with ControlNet Tile (faster, can flicker).")
            with gr.Row():
                width = gr.Slider(256, 2048, value=768, step=8, label="Width", info="Height follows the aspect ratio.")
                size_note = gr.Markdown(elem_classes=["adui-info"])

            with gr.Group(visible=True) as refine_box:
                with gr.Row():
                    context = gr.Slider(1, 32, value=16, step=1, label="Context frames",
                                        info="Lower this (e.g. 8) for very large sizes if you run out of VRAM.")
                    tile = gr.Slider(0, 1, value=0.75, step=0.05, label="Fidelity",
                                     info="How closely to stick to the source (ControlNet Tile strength). "
                                          "0.75 gives the most detail.")
                with gr.Row():
                    interp = gr.Slider(1, 4, value=1, step=1, label="Frame multiplier",
                                       interactive=rife_ok,
                                       info="Add in-between frames with RIFE for smoother motion." + ("" if rife_ok else " Install RIFE (Setup tab) to enable."))
                    repeats = gr.Slider(1, 4, value=1, step=1, label="Passes", info="Refine the result again N times.")

            with gr.Group(visible=False) as tile_box:
                with gr.Row():
                    strength = gr.Slider(0.1, 1, value=0.5, step=0.05, label="Strength",
                                         info="How much each frame may change.")
                    steps = gr.Slider(5, 60, value=20, step=1, label="Steps")
                with gr.Row():
                    use_ip2p = gr.Checkbox(label="+ Instruct-Pix2Pix", value=False, info="Helps keep the original colors.")
                    use_line = gr.Checkbox(label="+ Line art (anime)", value=False)

            with gr.Accordion("Change prompt or model", open=False):
                src_settings = gr.Markdown("Pick a video to see the settings it was made with.",
                                           elem_classes=["adui-info"])
                ckpt = gr.Dropdown([("Same as the source video", SAME)] + LIBRARY.choices("checkpoint"), value=SAME,
                                   label="Checkpoint", allow_custom_value=True,
                                   info="Optional. Leave fields empty to keep the source video's settings; "
                                        "changing them makes small edits, e.g. a different outfit.")
                head = gr.Textbox(label="Main prompt", lines=2, placeholder="(empty = the source video's prompt)")
                neg = gr.Textbox(label="Negative prompt", lines=2, placeholder="(empty = the source video's negative)")
                load_prompt = gr.Button("Copy source prompts", size="sm")

        with gr.Column(scale=2, min_width=380, elem_classes=["adui-sticky"]):
            run_btn = gr.Button("Upscale", variant="primary", elem_classes=["adui-generate"])
            issues = gr.Markdown(visible=False)
            panel = job_panel("upscale", "Upscaled video")

    source_outputs = [source, src_preview, src_info, width]

    def size_text(src, w):
        pngs = numbered_pngs(Path(src)) if src and Path(src).is_dir() else []
        if not pngs:
            return ""
        with Image.open(pngs[0]) as im:
            sw, sh = im.size
        h = int(int(w) * sh / sw) // 8 * 8
        return f"→ **{int(w)}×{h}** (from {sw}×{sh}, {w / sw:.2f}×)"

    def src_cfg(src) -> dict | None:
        cfgp = Path(src).parent / "prompt.json" if src else None
        return load_config_file(cfgp)[0] if cfgp and cfgp.is_file() else None

    def settings_md(src):
        cfg = src_cfg(src)
        if not cfg:
            return "Pick a video to see the settings it was made with."
        return (f"**Source settings:** {Path(str(cfg.get('path', ''))).name or '?'} · "
                f"prompt: *{(cfg.get('head_prompt') or '—')[:120]}*")

    def copy_prompts(src):
        cfg = src_cfg(src)
        if not cfg:
            raise gr.Error("Pick a source video first.")
        return cfg.get("head_prompt", ""), (cfg.get("n_prompt") or [""])[0]

    def on_source(src):
        preview, info, w = _src_info(src)
        # clear overrides so a previous video's edits never leak into this one
        return gr.update(value=preview, visible=bool(preview)), info, w, settings_md(src), SAME, "", ""

    source.change(on_source, [source], [src_preview, src_info, width, src_settings, ckpt, head, neg])
    # the dropdown shows the newest video on page load, and picking the same one later isn't a "change"
    demo.load(on_source, [source], [src_preview, src_info, width, src_settings, ckpt, head, neg])
    refresh.click(lambda: gr.update(choices=frame_dir_choices()), None, [source], queue=False)
    load_prompt.click(copy_prompts, [source], [head, neg])
    for c in (source, width):
        c.change(size_text, [source, width], [size_note], queue=False)
    method.change(lambda m: (gr.update(visible=m == REFINE), gr.update(visible=m == TILE)), [method],
                  [refine_box, tile_box], queue=False)

    def on_run(src, m, w, ctx, tile_s, mult, reps, stren, stp, ip2p, line, ck, hd, ng):
        errors = []
        frames = Path(src) if src else None
        if not frames or not frames.is_dir() or not numbered_pngs(frames):
            errors.append("Pick a frames folder that contains numbered PNGs.")
        cfg_src = frames.parent / "prompt.json" if frames else None
        if frames and not (cfg_src and cfg_src.is_file()):
            errors.append("No prompt.json found next to these frames, so the original settings are unknown.")
        ck = "" if ck == SAME else ck
        if ck and not LIBRARY.by_rel(ck):
            errors.append(f"Checkpoint not found: {ck}")
        if errors:
            return {issues: gr.update(value=issues_md(errors, []), visible=True)}

        cfg, _ = load_config_file(cfg_src)
        # overrides are opt-in: empty means "keep what the source video used"
        if ck:
            cfg["path"] = ck
        if hd and hd.strip():
            cfg["head_prompt"] = hd.strip()
        if ng and ng.strip():
            cfg["n_prompt"] = [ng.strip()]
        cfg["seed"] = [-1]
        cfg["img2img_map"]["enable"] = False      # refine re-renders from the tile ControlNet instead
        warns = []
        ck_info = LIBRARY.by_rel(cfg["path"])
        if ck_info and ck_info.arch == "sdxl" and m == TILE:
            return {issues: gr.update(value=issues_md(["Tile upscale does not support SDXL; use Refine."], []),
                                      visible=True)}
        if m == TILE:
            uc = cfg["upscale_config"]
            uc["strength"], uc["steps"] = float(stren), int(stp)
            uc["controlnet_tile"]["enable"] = True
            uc["controlnet_ip2p"]["enable"] = bool(ip2p)
            uc["controlnet_line_anime"]["enable"] = bool(line)
        if int(w) >= 1280 and m == REFINE and int(ctx) > 8:
            warns.append("Large refine sizes use a lot of VRAM; if it fails, lower Context frames to 8.")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        cfg_path = UI_CONFIG_DIR / f"{stamp}-{'refine' if m == REFINE else 'upscale'}-{cfg['name']}.json"
        UI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(dump(cfg), encoding="utf-8")

        n = len(numbered_pngs(frames))
        if m == REFINE:
            out_root = OUTPUT_DIRS["Refine"]
            argv = cli_argv("refine", frames, "-c", cfg_path, "-W", int(w), "-C", min(int(ctx), n),
                            "-t", float(tile_s), "-M", int(mult), "-r", int(reps), "-o", out_root)
            title = f"Refine · {frames.parent.name} → {int(w)}px"
        else:
            out_root = OUTPUT_DIRS["Upscale"]
            argv = cli_argv("tile-upscale", frames, "-c", cfg_path, "-W", int(w), "-o", out_root)
            title = f"Tile upscale · {frames.parent.name} → {int(w)}px"
        job = MANAGER.submit(Job(kind="upscale", title=title, argv=argv, out_root=out_root))
        return {issues: gr.update(value=issues_md([], warns), visible=bool(warns)), panel.job_id: job.id,
                panel.sig: None}

    run_btn.click(on_run, [source, method, width, context, tile, interp, repeats, strength, steps, use_ip2p,
                           use_line, ckpt, head, neg], [issues, panel.job_id, panel.sig])

    return SimpleNamespace(panel=panel, source_outputs=source_outputs, source=source)
