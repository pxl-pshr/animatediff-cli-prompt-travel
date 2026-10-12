"""Gallery tab: browse every finished run, replay it, reuse its settings, or delete it."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from core.media import numbered_pngs
from core.outputs import delete_run, list_runs, run_from_dir
from core.system import open_folder

CATEGORIES = ["All", "Generate", "Upscale", "Refine", "Stylize"]


def _thumb(run) -> str | None:
    if run.frame_dirs:
        pngs = numbered_pngs(run.frame_dirs[0])
        if pngs:
            return str(pngs[0])
    if run.preview and run.preview.suffix.lower() == ".gif":
        return str(run.preview)
    return None


def _items(category: str):
    runs = list_runs(None if category == "All" else [category])
    runs = [r for r in runs if _thumb(r)]
    items = [(_thumb(r), f"{r.when:%m-%d %H:%M} · {r.category} · {r.model or r.path.name}") for r in runs]
    return items, [str(r.path) for r in runs]


def _details(run_path: str | None):
    run = run_from_dir(Path(run_path)) if run_path else None
    if not run:
        return gr.update(value=None, visible=False), gr.update(value=None, visible=False), "Select a video."
    c = run.config
    loras = ", ".join(f"{Path(k).stem} ({v if not isinstance(v, dict) else 'regional'})"
                      for k, v in (c.get("lora_map") or {}).items()) or "none"
    pm = c.get("prompt_map") or {}
    timeline = "\n".join(f"| {k} | {str(v).replace('|', '/')[:140]} |" for k, v in list(pm.items())[:12])
    md = (f"### {run.when:%Y-%m-%d %H:%M} · {run.category}\n"
          f"**Model:** {run.model or '?'} · **Motion:** {Path(str(c.get('motion_module', ''))).stem or '?'} · "
          f"**Seed:** {run.seed} · **Size:** {run.size} · **Frames:** {run.frames}\n\n"
          f"**Sampler:** {c.get('scheduler', '?')} · **Steps:** {c.get('steps', '?')} · "
          f"**CFG:** {c.get('guidance_scale', '?')} · **LoRAs:** {loras}\n\n"
          f"**Main prompt:** {c.get('head_prompt') or '—'}\n\n"
          + (f"| Frame | Prompt |\n|---|---|\n{timeline}\n\n" if timeline else "")
          + f"**Negative:** {(c.get('n_prompt') or [''])[0] or '—'}\n\n`{run.rel}`")
    vid = run.preview
    if vid and vid.suffix.lower() == ".gif":
        return gr.update(value=None, visible=False), gr.update(value=str(vid), visible=True), md
    return gr.update(value=str(vid) if vid else None, visible=bool(vid)), gr.update(visible=False), md


def build(demo: gr.Blocks) -> SimpleNamespace:
    paths_state = gr.State([])
    selected = gr.State(None)

    with gr.Row():
        category = gr.Radio(CATEGORIES, value="All", label="Show", scale=4)
        refresh = gr.Button("Refresh", scale=1)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            gallery = gr.Gallery(label="Newest first", columns=5, height=640, object_fit="cover",
                                 allow_preview=False, show_label=True)
        with gr.Column(scale=2, elem_classes=["adui-sticky"]):
            video = gr.Video(label="Preview", visible=False, autoplay=True, loop=True, interactive=False, height=420)
            gif = gr.Image(label="Preview", visible=False, interactive=False, height=420)
            details = gr.Markdown("Select a video.")
            with gr.Row():
                load_settings = gr.Button("Load settings", size="sm")
                upscale_this = gr.Button("Upscale / refine", size="sm")
            with gr.Row():
                open_btn = gr.Button("Open folder", size="sm")
                confirm = gr.Checkbox(label="Confirm delete", value=False, scale=1)
                delete_btn = gr.Button("Delete", size="sm", variant="stop")

    def on_refresh(cat):
        items, paths = _items(cat)
        return items, paths

    category.change(on_refresh, [category], [gallery, paths_state])
    refresh.click(on_refresh, [category], [gallery, paths_state])
    demo.load(on_refresh, [category], [gallery, paths_state])

    def on_select(paths, evt: gr.SelectData):
        path = paths[evt.index] if evt.index is not None and evt.index < len(paths) else None
        return (path, *_details(path), False)

    gallery.select(on_select, [paths_state], [selected, video, gif, details, confirm])
    open_btn.click(lambda p: open_folder(p) if p else None, [selected], None, queue=False)

    def on_delete(path, ok, cat):
        if not path:
            raise gr.Error("Select a video first.")
        if not ok:
            raise gr.Error("Tick 'Yes, delete it' to confirm.")
        run = run_from_dir(Path(path))
        if run:
            delete_run(run)
            gr.Info("Deleted.")
        items, paths = _items(cat)
        return items, paths, None, gr.update(visible=False), gr.update(visible=False), "Select a video.", False

    delete_btn.click(on_delete, [selected, confirm, category],
                     [gallery, paths_state, selected, video, gif, details, confirm])

    return SimpleNamespace(selected=selected, upscale_this=upscale_this, load_settings=load_settings)
