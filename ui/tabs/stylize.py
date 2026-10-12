"""Video → Video tab: a 3-step wizard around `animatediff stylize create-config / generate`."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from components import help_label, issues_md, job_panel
from core import build as B
from core.config import IP_ADAPTER_VARIANTS, SCHEDULERS, default_config, dump, ip_variant_flags, \
    ip_variant_of, normalize
from core.jobs import MANAGER, Job, cli_argv
from core.media import is_video, numbered_pngs, probe_video, save_image_at_frame, spread_frames
from core.paths import STYLIZE_DIR, UI_CONFIG_DIR, UI_INPUT_DIR, from_data_rel, to_data_rel
from core.system import features
from core.validate import check_generate
from tabs.txt2vid import ckpt_choices, lora_choices, mm_choices

# ControlNet guides offered for restyling, with their defaults (enabled, strength).
GUIDES = [
    ("controlnet_openpose", "Pose", True, 1.0),
    ("controlnet_ip2p", "Instruct-Pix2Pix (keep the scene)", True, 0.5),
    ("controlnet_softedge", "Soft edge", False, 0.8),
    ("controlnet_lineart", "Line art", False, 0.8),
    ("controlnet_lineart_anime", "Line art (anime)", False, 0.8),
    ("controlnet_depth", "Depth", False, 0.8),
    ("controlnet_canny", "Canny edges", False, 0.6),
    ("controlnet_tile", "Tile (keep colors & details)", False, 0.6),
    ("animatediff_controlnet", "AnimateDiff ControlNet (replace characters)", False, 1.0),
]
UPSCALE_FACTORS = [("No second pass", 0.0), ("1.25×", 1.25), ("1.5×", 1.5), ("2×", 2.0)]


def project_choices():
    items = []
    if STYLIZE_DIR.is_dir():
        for d in sorted(STYLIZE_DIR.iterdir(), reverse=True):
            cfg = d / "prompt.json"
            if d.is_dir() and cfg.is_file():
                n = len(numbered_pngs(d / "00_img2img"))
                items.append((f"{d.name} · {n} frames", str(d)))
    return items


def _read_cfg(project: Path) -> dict:
    return json.loads((project / "prompt.json").read_text(encoding="utf-8"))


def save_project(path, *vals, ip_images=None) -> tuple[Path, dict, list[str], list[str]]:
    if not path or not (Path(path) / "prompt.json").is_file():
        raise gr.Error("Pick a project first (step ②).")
    proj = Path(path)
    raw = _read_cfg(proj)
    cfg, _ = normalize(raw)
    it = iter(vals)
    ck, mm, head, timeline, tail, neg = (next(it) for _ in range(6))
    l0, l0w, l1, l1w = (next(it) for _ in range(4))
    seed, steps, cfg_scale, sched, i2i, i2i_s = (next(it) for _ in range(6))
    boxes = [next(it) for _ in GUIDES]
    weights = [next(it) for _ in GUIDES]
    ip_on, ip_var, ip_scale = next(it), next(it), next(it)
    w, h, ctx, up, up_tile, hires, fmt = (next(it) for _ in range(7))

    cfg.update({"path": ck or "", "motion_module": mm or "", "head_prompt": (head or "").strip(),
                "prompt_map": B.timeline_to_map(timeline), "tail_prompt": (tail or "").strip(),
                "n_prompt": [(neg or "").strip()], "seed": [int(seed) if seed is not None else -1],
                "steps": int(steps), "guidance_scale": float(cfg_scale), "scheduler": sched})
    cfg["lora_map"], adapters = B.split_loras(((l0, l0w), (l1, l1w)))
    cams = {r: w for r, w in (cfg.get("motion_lora_map") or {}).items() if not B.is_adapter(r)}
    cfg["motion_lora_map"] = {**cams, **adapters}
    cfg["img2img_map"].update({"enable": bool(i2i), "denoising_strength": float(i2i_s)})

    cn = cfg["controlnet_map"]
    cn_root = from_data_rel(cn.get("input_image_dir", "")) if cn.get("input_image_dir") else proj / "00_controlnet_image"
    cn["input_image_dir"] = to_data_rel(cn_root)
    frames_dir = proj / "00_img2img"
    for (key, *_), on, weight in zip(GUIDES, boxes, weights):
        unit = cn.setdefault(key, {})
        unit.update({"enable": bool(on), "controlnet_conditioning_scale": float(weight)})
        for k, dv in (("use_preprocessor", True), ("guess_mode", False), ("control_guidance_start", 0.0),
                      ("control_guidance_end", 1.0), ("control_scale_list", []), ("control_region_list", [])):
            unit.setdefault(k, dv)
        folder = cn_root / key
        if on and not numbered_pngs(folder):
            # every guide reads the source frames from its own folder
            folder.mkdir(parents=True, exist_ok=True)
            for p in numbered_pngs(frames_dir):
                shutil.copy2(p, folder / p.name)

    ip = cfg["ip_adapter_map"]
    ip_dir = from_data_rel(ip.get("input_image_dir", "")) if ip.get("input_image_dir") else proj / "00_ipadapter"
    imgs = B._gallery_paths(ip_images)
    if imgs:
        shutil.rmtree(ip_dir, ignore_errors=True)
        n = len(numbered_pngs(frames_dir)) or 1
        for p, f in zip(imgs, spread_frames(len(imgs), n)):
            save_image_at_frame(p, ip_dir, f)
        ip_on = True
    ip.update({"enable": bool(ip_on) and bool(numbered_pngs(ip_dir)), "input_image_dir": to_data_rel(ip_dir),
               "scale": float(ip_scale), **ip_variant_flags(ip_var)})

    sc = cfg.setdefault("stylize_config", {})
    p0 = sc.setdefault("0", {})
    ctx = int(ctx)
    p0.update({"width": int(w) // 8 * 8, "height": int(h) // 8 * 8, "context": ctx, "overlap": ctx // 4})
    p0.setdefault("stride", 0)
    p0.setdefault("length", len(numbered_pngs(frames_dir)))
    if up and float(up) > 0:
        p1 = sc.setdefault("1", {})
        tmpl = {"controlnet_tile": {"enable": True, "use_preprocessor": True, "guess_mode": False,
                                    "controlnet_conditioning_scale": 1.0, "control_guidance_start": 0.0,
                                    "control_guidance_end": 1.0, "control_scale_list": []},
                "controlnet_ip2p": {"enable": False, "use_preprocessor": True, "guess_mode": False,
                                    "controlnet_conditioning_scale": 0.5, "control_guidance_start": 0.0,
                                    "control_guidance_end": 1.0, "control_scale_list": []},
                "ip_adapter": False, "reference": False, "img2img": False, "interpolation_multiplier": 1,
                "stride": 0}
        for k, dv in tmpl.items():
            p1.setdefault(k, dv)
        p1.update({"width": int(int(w) * float(up)) // 8 * 8, "height": int(int(h) * float(up)) // 8 * 8,
                   "length": p0["length"], "context": 8, "overlap": 2, "steps": int(steps),
                   "guidance_scale": float(cfg_scale)})
        p1["controlnet_tile"]["controlnet_conditioning_scale"] = float(up_tile)
    else:
        sc.pop("1", None)
    cfg["gradual_latent_hires_fix_map"]["enable"] = bool(hires)
    cfg["output"] = {**(cfg.get("output") or {}), "format": fmt}
    cfg["output"].setdefault("fps", 8)

    errors, warns = check_generate(cfg, width=p0["width"], height=p0["height"],
                                   length=p0["length"], context=min(ctx, p0["length"]))
    # the stylize tool writes its own frames; don't flag prompt keyframes past a test-run length
    (proj / "prompt.json").write_text(dump(cfg), encoding="utf-8")
    return proj, cfg, errors, warns


def build(demo: gr.Blocks) -> SimpleNamespace:
    ok, how = features()["Video → Video (stylize)"]
    if not ok:
        gr.Markdown(f"⚠️ Video → Video needs a few extra packages. Install them with:\n\n`{how}`",
                    elem_classes=["adui-warn"])

    has_projects = bool(project_choices())
    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=520):
            # ============================================================ project
            with gr.Group():
                with gr.Row():
                    project = gr.Dropdown(project_choices(), label="Project", scale=6, allow_custom_value=True,
                                          info="A prepared video. Create one below from any video file.")
                    proj_refresh = gr.Button("🔄", scale=0, min_width=44)
                proj_info = gr.Markdown("",
                                        elem_classes=["adui-info"])

            with gr.Accordion("New project from a video", open=not has_projects):
                src_video = gr.Video(label="Video", sources=["upload"], height=240)
                src_info = gr.Markdown(elem_classes=["adui-info"])
                with gr.Row():
                    fps = gr.Slider(4, 30, value=8, step=1, label="FPS",
                                    info="Frames per second to take from the video. 8–12 is typical; "
                                         "more = longer render.")
                    short_edge = gr.Slider(256, 1024, value=512, step=8, label="Size", info="Short side of the frames, in pixels.")
                with gr.Row():
                    offset = gr.Number(value=0, precision=0, minimum=0, label="Start (s)")
                    duration = gr.Number(value=0, precision=0, minimum=0, label="Length (s)", info="0 = until the end.")
                    tag_every = gr.Slider(1, 32, value=8, step=1, label="Describe every",
                                          info="Auto-describe every N frames (WD14 tagger). Each becomes a timeline row.")
                p_lowvram = gr.Checkbox(label="Low VRAM", value=False, info="For cards under 12 GB.")
                prep_btn = gr.Button("Create project", variant="secondary")

            # ============================================================ style
            with gr.Group():
                with gr.Row():
                    s_ckpt = gr.Dropdown(ckpt_choices(), label="Checkpoint", allow_custom_value=True, scale=3)
                    s_mm = gr.Dropdown(mm_choices(), label="Motion module", allow_custom_value=True, scale=3)
                s_head = gr.Textbox(label="Style prompt", lines=2, info="Added to every frame.",
                                    placeholder="masterpiece, best quality, anime style, cel shading")
                help_label("Timeline", "Auto-descriptions of the video, one row per N frames. Edit freely. The "
                                       "describer is trained on anime art and can invent tags on real footage "
                                       "(e.g. '1girl'); clear them if they don't match.")
                s_timeline = gr.Dataframe(headers=["Frame", "Description"], show_label=False,
                                          datatype=["number", "str"], col_count=(2, "fixed"),
                                          row_count=(1, "dynamic"), type="array", interactive=True, wrap=True,
                                          label="Timeline", column_widths=["14%", "86%"], max_height=300)
                with gr.Row():
                    clear_desc = gr.Button("Clear descriptions", size="sm")
                s_tail = gr.Textbox(label="Ending prompt", lines=1, max_lines=1, info="Optional. Added after every row.")
                s_neg = gr.Textbox(label="Negative prompt", lines=2)
                with gr.Row():
                    s_lora0 = gr.Dropdown(lora_choices(), value="", label="LoRA", allow_custom_value=True, scale=3)
                    s_lora0_w = gr.Slider(-1, 2, value=0.8, step=0.05, label="Weight", scale=2)
                    s_lora1 = gr.Dropdown(lora_choices(), value="", label="LoRA", allow_custom_value=True, scale=3)
                    s_lora1_w = gr.Slider(-1, 2, value=0.8, step=0.05, label="Weight", scale=2)
                with gr.Row():
                    s_seed = gr.Number(value=-1, precision=0, label="Seed", minimum=-1)
                    s_steps = gr.Slider(1, 60, value=20, step=1, label="Steps")
                    s_cfg = gr.Slider(1, 20, value=8, step=0.5, label="CFG")
                    s_sched = gr.Dropdown(SCHEDULERS, value="k_dpmpp_sde", label="Sampler")

            with gr.Accordion("Follow the source", open=True):
                s_i2i = gr.Checkbox(label="Start from the source frames", value=True,
                                    info="img2img: begin from the original frames instead of noise.")
                s_i2i_strength = gr.Slider(0.1, 1, value=0.7, step=0.05, label="Change",
                                           info="How much to change the frames. Lower keeps more of the original.")
                guide_boxes, guide_weights = [], []
                for key, label, on, w in GUIDES:
                    with gr.Row():
                        guide_boxes.append(gr.Checkbox(label=label, value=on, scale=3))
                        guide_weights.append(gr.Slider(0, 2, value=w, step=0.05, label="Strength", scale=2))

            with gr.Accordion("Match a look or character", open=False):
                s_ip_enable = gr.Checkbox(label="Reference images", value=False,
                                          info="IP-Adapter: steer the look or keep a character / face.")
                s_ip_images = gr.Gallery(type="filepath", interactive=True, columns=5, height=160,
                                         file_types=["image"], label="Reference images")
                with gr.Row():
                    s_ip_variant = gr.Dropdown(IP_ADAPTER_VARIANTS, value="plus", label="Adapter type")
                    s_ip_scale = gr.Slider(0, 1, value=0.5, step=0.05, label="Strength")

            with gr.Accordion("Size & second pass", open=False):
                with gr.Row():
                    s_width = gr.Slider(256, 1536, value=512, step=8, label="Width")
                    s_height = gr.Slider(256, 1536, value=512, step=8, label="Height")
                    s_context = gr.Slider(4, 32, value=16, step=1, label="Context frames")
                with gr.Row():
                    s_up = gr.Dropdown(UPSCALE_FACTORS, value=1.5, label="Second pass",
                                   info="Render again at a larger size to add detail and clean up.")
                    s_up_tile = gr.Slider(0, 1, value=1.0, step=0.05, label="Second pass fidelity",
                                       info="How closely the second pass sticks to the first.")
                s_hires = gr.Checkbox(label="Hi-res fix", value=False,
                                      info="Only for 768 px and up; use the Euler a sampler.")
            mask_note = features()["Masks / regions / composite"]
            gr.Markdown(
                "**Separate subject & background (masks / regions):** "
                + ("available from the command line: `animatediff stylize create-mask` / `create-region`."
                   if mask_note[0] else f"needs extra packages (`{mask_note[1]}`)."),
                elem_classes=["adui-info"])

        # ================================================================ right: render + result
        with gr.Column(scale=2, min_width=380, elem_classes=["adui-sticky"]):
            full_btn = gr.Button("Render", variant="primary", elem_classes=["adui-generate"])
            with gr.Row():
                test_btn = gr.Button("Test run", variant="secondary", scale=2)
                test_len = gr.Number(value=16, precision=0, minimum=8, maximum=64, label="Frames", scale=1,
                                     min_width=80)
                test_from = gr.Number(value=0, precision=0, minimum=0, label="From", scale=1, min_width=80)
            with gr.Row():
                s_fmt = gr.Radio(["mp4", "gif", "webm"], value="mp4", label="Format", scale=3)
                save_btn = gr.Button("Save", size="sm", scale=1)
            issues = gr.Markdown(visible=False)
            preview_img = gr.Image(label="First frame", interactive=False, height=220, visible=False)

            def after_prep(job: Job):
                if job.meta.get("step") == "prep" and job.save_dir:
                    return [gr.update(choices=project_choices(), value=str(job.save_dir))]
                return [gr.skip()]

            panel = job_panel("stylize", "Stylized video", extra_outputs=[project], on_done=after_prep)
            with gr.Row(visible=False) as result_actions:
                upscale_this = gr.Button("Upscale / refine this", size="sm")

    style_inputs = [s_ckpt, s_mm, s_head, s_timeline, s_tail, s_neg, s_lora0, s_lora0_w, s_lora1, s_lora1_w,
                    s_seed, s_steps, s_cfg, s_sched, s_i2i, s_i2i_strength, *guide_boxes, *guide_weights,
                    s_ip_enable, s_ip_variant, s_ip_scale, s_width, s_height, s_context, s_up, s_up_tile, s_hires, s_fmt]

    # ------------------------------------------------------------------ step 1
    def on_video(path):
        if not path:
            return ""
        try:
            info = probe_video(path)
            return (f"{info.width}×{info.height}, {info.fps:.2f} fps, {info.duration:.1f} s "
                    f"(≈ {int(info.duration * 8)} frames at 8 fps)")
        except Exception as e:
            return f"⚠️ Could not read this video: {e}"

    src_video.change(on_video, [src_video], [src_info])

    def on_prepare(video, fps_, edge, off, dur, every, ck, mm, i2i, lowvram, hires):
        errors = []
        if not video or not is_video(video):
            errors.append("Upload a source video.")
        if not ck:
            errors.append("Pick a checkpoint.")
        if not mm:
            errors.append("Pick a motion module.")
        if errors:
            return {issues: gr.update(value=issues_md(errors, []), visible=True)}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        keep = UI_INPUT_DIR / f"{stamp}-stylize-source"
        keep.mkdir(parents=True, exist_ok=True)
        src = keep / f"source{Path(video).suffix.lower()}"
        shutil.copy2(video, src)

        base = default_config()
        base.update({"name": "stylize", "path": ck, "motion_module": mm})
        if hires:
            base["scheduler"] = "euler_a"   # the gradual hires fix is built around Euler a
        UI_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        base_path = UI_CONFIG_DIR / f"{stamp}-stylize-base.json"
        base_path.write_text(dump(base), encoding="utf-8")

        args = ["stylize", "create-config", src, "-c", base_path, "-f", int(fps_), "-sh", int(edge),
                "-of", int(off or 0), "-d", int(dur) if dur and dur > 0 else -1, "-p", int(every), "-o", STYLIZE_DIR]
        if i2i:
            args.append("-i2i")
        if lowvram:
            args.append("-lo")
        if hires:
            args.append("-gh")
        job = MANAGER.submit(Job(kind="stylize", title=f"Prepare · {Path(video).name}", argv=cli_argv(*args),
                                 out_root=STYLIZE_DIR, meta={"step": "prep"}))
        return {issues: gr.update(visible=False), panel.job_id: job.id, panel.sig: None}

    prep_btn.click(on_prepare, [src_video, fps, short_edge, offset, duration, tag_every, s_ckpt, s_mm, s_i2i,
                                p_lowvram, s_hires], [issues, panel.job_id, panel.sig])

    # ------------------------------------------------------------------ step 2: load / save
    def on_project(path):
        if not path or not (Path(path) / "prompt.json").is_file():
            return [gr.skip()] * (len(style_inputs) + 2)
        proj = Path(path)
        cfg, _ = normalize(_read_cfg(proj))
        frames = numbered_pngs(proj / "00_img2img")
        sc = cfg.get("stylize_config") or {}
        p0 = sc.get("0", {})
        p1 = sc.get("1")
        cn = cfg["controlnet_map"]
        vals = [
            cfg.get("path") or None, cfg.get("motion_module") or None, cfg.get("head_prompt", ""),
            B.map_to_timeline(cfg.get("prompt_map")), cfg.get("tail_prompt", ""), (cfg.get("n_prompt") or [""])[0],
        ]
        loras = (list((cfg.get("lora_map") or {}).items())
                 + [(r, w) for r, w in (cfg.get("motion_lora_map") or {}).items() if B.is_adapter(r)])[:2]
        loras += [("", 0.8)] * 2
        for rel, w in loras[:2]:
            vals += [rel, float(w) if not isinstance(w, dict) else 0.8]
        vals += [int((cfg.get("seed") or [-1])[0]), int(cfg.get("steps", 20)), float(cfg.get("guidance_scale", 8)),
                 cfg.get("scheduler", "k_dpmpp_sde"), bool(cfg["img2img_map"].get("enable")),
                 float(cfg["img2img_map"].get("denoising_strength", 0.7))]
        vals += [bool((cn.get(k) or {}).get("enable")) for k, *_ in GUIDES]
        vals += [float((cn.get(k) or {}).get("controlnet_conditioning_scale", w)) for k, _, _, w in GUIDES]
        ip = cfg["ip_adapter_map"]
        ip_dir = from_data_rel(ip.get("input_image_dir", "")) if ip.get("input_image_dir") else None
        vals += [bool(ip.get("enable")) and bool(ip_dir and numbered_pngs(ip_dir)), ip_variant_of(ip),
                 float(ip.get("scale", 0.5))]
        w0, h0 = int(p0.get("width", 512)), int(p0.get("height", 512))
        factor = 0.0
        if p1:
            factor = min((f for _, f in UPSCALE_FACTORS if f), key=lambda f: abs(f - p1.get("width", w0) / w0))
        vals += [w0, h0, int(p0.get("context", 16)), factor,
                 float((p1 or {}).get("controlnet_tile", {}).get("controlnet_conditioning_scale", 1.0)),
                 bool((cfg.get("gradual_latent_hires_fix_map") or {}).get("enable")),
                 (cfg.get("output") or {}).get("format", "mp4")]
        info = (f"**{len(frames)} frames** · {w0}×{h0} · {(cfg.get('output') or {}).get('fps', 8)} fps · "
                f"`{proj.name}`")
        first = str(frames[0]) if frames else None
        return vals + [info, gr.update(value=first, visible=bool(first))]

    project.change(on_project, [project], style_inputs + [proj_info, preview_img])
    clear_desc.click(lambda: [[0, ""]], None, [s_timeline], queue=False)
    proj_refresh.click(lambda: gr.update(choices=project_choices()), None, [project], queue=False)

    def on_save(path, *vals):
        ip_images = vals[-1]
        _, _, errors, warns = save_project(path, *vals[:-1], ip_images=ip_images)
        gr.Info("Settings saved.")
        return gr.update(value=issues_md(errors, warns), visible=bool(errors or warns))

    save_btn.click(on_save, [project] + style_inputs + [s_ip_images], [issues])

    # ------------------------------------------------------------------ step 3
    def render(path, test, length, start, *vals):
        ip_images = vals[-1]
        proj, cfg, errors, warns = save_project(path, *vals[:-1], ip_images=ip_images)
        if errors:
            return {issues: gr.update(value=issues_md(errors, warns), visible=True)}
        args = ["stylize", "generate", proj]
        total = cfg["stylize_config"]["0"]["length"]
        if test:
            start = max(0, min(int(start or 0), total - 1))
            args += ["-L", int(length), "-FO", start]
            title = f"Test run · {proj.name} · frames {start}–{start + int(length) - 1}"
        else:
            title = f"Render · {proj.name} · {total} frames"
        job = MANAGER.submit(Job(kind="stylize", title=title, argv=cli_argv(*args), out_root=proj,
                                 meta={"step": "render", "result_root": str(proj)}))
        return {issues: gr.update(value=issues_md([], warns), visible=bool(warns)), panel.job_id: job.id,
                panel.sig: None}

    test_btn.click(lambda p, l, s, *v: render(p, True, l, s, *v), [project, test_len, test_from] + style_inputs
                   + [s_ip_images], [issues, panel.job_id, panel.sig])
    full_btn.click(lambda p, l, s, *v: render(p, False, l, s, *v), [project, test_len, test_from] + style_inputs
                   + [s_ip_images], [issues, panel.job_id, panel.sig])

    def on_page(path):
        # the dropdown shows the newest project on page load, but .change doesn't fire for it
        if path and (Path(path) / "prompt.json").is_file():
            return on_project(path)
        ck = ckpt_choices()
        mms = mm_choices()
        pick = next((v for l, v in mms if "v15_v2" in v), mms[0][1] if mms else None)
        out = [gr.skip()] * (len(style_inputs) + 2)
        out[0], out[1] = (ck[0][1] if ck else None), pick
        return out

    demo.load(on_page, [project], style_inputs + [proj_info, preview_img])
    panel.run_path.change(lambda p: gr.update(visible=bool(p)), [panel.run_path], [result_actions], queue=False)
    return SimpleNamespace(panel=panel, upscale_this=upscale_this)
