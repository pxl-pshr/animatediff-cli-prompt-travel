"""AnimateDiff Prompt Travel — web UI.

Run with webui.bat (Windows) or: ui/.venv/Scripts/python ui/app.py [--port 7860] [--share] [--listen]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

import html  # noqa: E402

from components import CSS  # noqa: E402
from core.jobs import MANAGER, QUEUED  # noqa: E402
from core.library import LIBRARY  # noqa: E402
from core.paths import DATA_DIR, OUTPUT_DIRS, REPO_ROOT, UI_STATE_DIR, cli_python, ensure_dirs  # noqa: E402
from tabs import gallery, setup, stylize, txt2vid, upscale  # noqa: E402

THEME = gr.themes.Default(
    primary_hue="violet", secondary_hue="zinc", neutral_hue="zinc", radius_size="md", text_size="sm",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    block_shadow="none", block_title_text_weight="500", block_label_text_weight="500",
    button_primary_shadow="none", button_secondary_shadow="none", button_large_text_weight="600",
)

HEADER = "<div style='font-size:1.15em;font-weight:600;letter-spacing:-.01em;padding:4px 0'>AnimateDiff Prompt Travel</div>"


def global_status() -> str:
    """Header pill showing the running job (any tab) and the queue length."""
    cur = MANAGER.current()
    queued = sum(1 for j in MANAGER.jobs() if j.status == QUEUED)
    if not cur and not queued:
        return "<div class='adui-global idle'>Idle</div>"
    text = f"● {html.escape(cur.title)} — {html.escape(cur.status_line())}" if cur else "Starting…"
    if queued:
        text += f" · {queued} queued"
    return f"<div class='adui-global busy'>{text}</div>"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AnimateDiff Prompt Travel", theme=THEME, css=CSS, analytics_enabled=False) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                gr.HTML(HEADER)
            with gr.Column(scale=2, min_width=240):
                status = gr.HTML(global_status())
        gr.Timer(2.0).tick(global_status, None, status, show_progress="hidden", queue=False)
        with gr.Tabs() as tabs:
            with gr.Tab("Text → Video", id="txt2vid"):
                t2v = txt2vid.build(demo)
            with gr.Tab("Video → Video", id="stylize"):
                sty = stylize.build(demo)
            with gr.Tab("Upscale & Refine", id="upscale"):
                up = upscale.build(demo)
            with gr.Tab("Gallery", id="gallery"):
                gal = gallery.build(demo)
            with gr.Tab("Setup & Models", id="setup"):
                setup.build(demo)

        # ---- cross-tab wiring: "Upscale this" buttons jump to the Upscale tab with the source filled in
        def to_upscale(run_path):
            src = upscale.source_for_run(run_path)
            if not src:
                raise gr.Error("Nothing to upscale yet — generate (or pick) a video first.")
            return gr.Tabs(selected="upscale"), *upscale.select_source(src)

        up_outputs = [tabs] + up.source_outputs
        t2v.upscale_this.click(to_upscale, [t2v.panel.run_path], up_outputs)
        sty.upscale_this.click(to_upscale, [sty.panel.run_path], up_outputs)
        gal.upscale_this.click(to_upscale, [gal.selected], up_outputs)

        def to_txt2vid(run_path):
            if not run_path:
                raise gr.Error("Select a video first.")
            out = t2v.load_into_form(f"run:{run_path}")
            out[tabs] = gr.Tabs(selected="txt2vid")
            return out

        gal.load_settings.click(to_txt2vid, [gal.selected], [tabs] + t2v.load_outputs)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="AnimateDiff Prompt Travel web UI")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ADUI_PORT", 7860)))
    parser.add_argument("--listen", action="store_true", help="Listen on all interfaces (LAN access)")
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    args = parser.parse_args()

    ensure_dirs()
    if not cli_python().exists():
        print(f"[webui] WARNING: main venv not found at {cli_python()} — generation will fail until it is installed.")
    print("[webui] Scanning models ...")
    LIBRARY.scan()

    demo = build_app()
    demo.queue(default_concurrency_limit=4, max_size=64)
    allowed = [str(p) for p in OUTPUT_DIRS.values()] + [str(DATA_DIR), str(UI_STATE_DIR)]
    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        allowed_paths=allowed,
        show_api=False,
        favicon_path=None,
    )


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
