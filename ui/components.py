"""Widgets shared by several tabs."""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gradio as gr

from core.jobs import CANCELLED, DONE, FAILED, MANAGER, QUEUED, RUNNING, Job
from core.outputs import Run, run_from_dir
from core.system import open_folder


def progress_html(job: Job | None) -> str:
    if job is None:
        return "<div class='adui-progress idle'><div class='adui-status'>Ready.</div></div>"
    pct = 0
    cls = job.status
    if job.status == RUNNING:
        pct = int(job.fraction * 100) if job.total else 0
    elif job.status == DONE:
        pct = 100
    status = html.escape(job.status_line())
    if job.status == QUEUED:
        pos = MANAGER.queue_position(job)
        status = f"Queued (position {pos}) — waits for the current job to finish." if pos else status
    bar = "" if job.status in (FAILED, CANCELLED) else (
        f"<div class='adui-bar'><div class='adui-fill{' indeterminate' if job.status == RUNNING and not job.total else ''}' "
        f"style='width:{pct if job.total or job.status != RUNNING else 100}%'></div></div>")
    title = html.escape(job.title)
    return (f"<div class='adui-progress {cls}'><div class='adui-title'>{title}</div>{bar}"
            f"<div class='adui-status'>{status}</div></div>")


def error_md(job: Job | None) -> str:
    if not job or job.status != FAILED:
        return ""
    hint = f"\n\n**How to fix:** {job.hint}" if job.hint else ""
    return f"**{job.error}**{hint}"


def resolve_run(job: Job) -> Run | None:
    root = job.meta.get("result_root")
    # never fall back to the shared output folder: that would show an older run's video
    for cand in (Path(root) if root else None, job.save_dir):
        if cand and Path(cand).is_dir():
            run = run_from_dir(Path(cand))
            if run:
                return run
    return None


@dataclass
class JobPanel:
    job_id: gr.State
    sig: gr.State
    progress: gr.HTML
    cancel: gr.Button
    open_dir: gr.Button
    video: gr.Video
    gif: gr.Image
    error: gr.Markdown
    log: gr.Textbox
    run_path: gr.State

    @property
    def outputs(self) -> list:
        return [self.progress, self.cancel, self.open_dir, self.video, self.gif, self.error, self.log,
                self.sig, self.run_path]


def job_panel(kind: str, result_label: str = "Result", extra_outputs: list | None = None,
              on_done: Callable[[Job], list] | None = None) -> JobPanel:
    """Progress/result panel that follows the newest job of `kind` (or the one in job_id).

    on_done(job) -> values for extra_outputs, called once when a job finishes successfully.
    """
    extra_outputs = extra_outputs or []
    job_id = gr.State(None)
    sig = gr.State(None)
    run_path = gr.State(None)
    progress = gr.HTML(progress_html(None))
    with gr.Row():
        cancel = gr.Button("Cancel", variant="stop", size="sm", visible=False)
        open_dir = gr.Button("Open folder", size="sm", visible=False)
    error = gr.Markdown(visible=False, elem_classes=["adui-error"])
    video = gr.Video(label=result_label, visible=False, autoplay=True, loop=True, interactive=False, height=460)
    gif = gr.Image(label=result_label, visible=False, type="filepath", interactive=False, height=460)
    with gr.Accordion("Log", open=False):
        log = gr.Textbox(show_label=False, lines=16, max_lines=16, autoscroll=True, interactive=False,
                         elem_classes=["adui-log"])
    panel = JobPanel(job_id, sig, progress, cancel, open_dir, video, gif, error, log, run_path)

    n_out = 9 + len(extra_outputs)

    def poll(jid, old_sig):
        job = MANAGER.get(jid) or MANAGER.latest(kind)
        if job is None:
            return [gr.skip()] * n_out
        new_sig = (job.id, job.status, job.stage, job.step, job.total, len(job.lines),
                   int(job.elapsed) if job.status == RUNNING else 0)
        if new_sig == old_sig:
            return [gr.skip()] * n_out
        active = job.status in (QUEUED, RUNNING)
        changed_state = old_sig is None or old_sig[0] != job.id or old_sig[1] != job.status
        video_u = gif_u = gr.skip()
        run_u = gr.skip()
        if changed_state:
            video_u, gif_u = gr.update(visible=False), gr.update(visible=False)
            run_u = None
            if job.status == DONE:
                run = resolve_run(job)
                if run and run.preview:
                    run_u = str(run.path)
                    if run.preview.suffix.lower() == ".gif":
                        gif_u = gr.update(value=str(run.preview), visible=True)
                    else:
                        video_u = gr.update(value=str(run.preview), visible=True)
                elif run:
                    run_u = str(run.path)
        err = error_md(job)
        extras = [gr.skip()] * len(extra_outputs)
        # only fire on a live transition, not when a freshly opened page first sees an old job
        if on_done and changed_state and old_sig is not None and job.status == DONE:
            try:
                extras = list(on_done(job))
            except Exception as e:
                print(f"[webui] on_done hook failed: {e!r}")
        return [
            progress_html(job),
            gr.update(visible=active),
            gr.update(visible=job.status == DONE or bool(job.save_dir)),
            video_u,
            gif_u,
            gr.update(value=err, visible=bool(err)),
            job.tail(),
            new_sig,
            run_u,
        ] + extras

    def do_cancel(jid):
        job = MANAGER.get(jid) or MANAGER.latest(kind)
        if job:
            MANAGER.cancel(job.id)

    def do_open(jid, rp):
        job = MANAGER.get(jid) or MANAGER.latest(kind)
        target = rp or (job and (job.save_dir or job.out_root))
        if target:
            open_folder(target)

    timer = gr.Timer(1.0)
    timer.tick(poll, [job_id, sig], panel.outputs + extra_outputs, show_progress="hidden", queue=False)
    cancel.click(do_cancel, [job_id], None, queue=False)
    open_dir.click(do_open, [job_id, run_path], None, queue=False)
    return panel


def help_label(text: str, tip: str) -> gr.HTML:
    """A field label with a '?' tooltip, for components that have no `info=` (e.g. Dataframe)."""
    return gr.HTML(f"<div class='adui-label'>{html.escape(text)}"
                   f"<span class='adui-q' tabindex='0' data-tip='{html.escape(tip, quote=True)}'>?</span></div>")


def info_box(text: str, kind: str = "info"):
    return gr.Markdown(text, elem_classes=[f"adui-{kind}"])


def issues_md(errors: list[str], warnings: list[str], notes: list[str] | None = None) -> str:
    parts = []
    if errors:
        parts.append("**Fix these before generating:**\n" + "\n".join(f"- ❌ {e}" for e in errors))
    if warnings:
        parts.append("**Heads up:**\n" + "\n".join(f"- ⚠️ {w}" for w in warnings))
    if notes:
        parts.append("\n".join(f"- ℹ️ {n}" for n in notes))
    return "\n\n".join(parts)


CSS = """
.adui-progress {border:1px solid var(--border-color-primary); border-radius:10px; padding:12px 14px;
  background:var(--block-background-fill);}
.adui-progress .adui-title {font-weight:600; margin-bottom:8px; font-size:0.95em;}
.adui-progress .adui-status {font-size:0.9em; opacity:0.85; margin-top:6px;}
.adui-bar {height:10px; border-radius:6px; background:var(--neutral-200); overflow:hidden;}
.dark .adui-bar {background:var(--neutral-700);}
.adui-fill {height:100%; background:var(--color-accent); transition:width .6s ease;}
.adui-fill.indeterminate {background:linear-gradient(90deg, transparent, var(--color-accent), transparent);
  background-size:200% 100%; animation:adui-slide 1.4s linear infinite;}
@keyframes adui-slide {from {background-position:200% 0} to {background-position:-200% 0}}
.adui-progress.failed {border-color:var(--error-border-color, #e5484d);}
.adui-progress.done {border-color:var(--color-accent);}
.adui-error {border-left:4px solid #e5484d; padding:6px 12px !important; border-radius:6px;
  background:rgba(229,72,77,.08);}
.adui-warn {border-left:4px solid #f5a524; padding:6px 12px !important; border-radius:6px;
  background:rgba(245,165,36,.08);}
.adui-info {opacity:.85; font-size:.92em;}
.adui-log textarea {font-family:ui-monospace, Consolas, monospace !important; font-size:12px !important;}
.adui-generate {min-height:56px; font-size:1.15em !important;}
.adui-sticky {position:sticky; top:12px;}
/* ---- help text (info=) shown as a '?' badge with a hover tooltip instead of a line under the label ---- */
span[data-testid="block-info"].has-info::after,
.block > label:has(> input[data-testid="checkbox"]):has(+ div) > span:last-child::after,
.adui-q {
  content:"?"; display:inline-flex; align-items:center; justify-content:center; width:14px; height:14px;
  margin-left:6px; border-radius:50%; font-size:10px; font-weight:600; line-height:1; vertical-align:1px;
  color:var(--body-text-color-subdued); border:1px solid var(--border-color-primary); cursor:help; }
span[data-testid="block-info"].has-info + div,
.block > label:has(> input[data-testid="checkbox"]) + div,
.adui-q::after {
  display:none; position:absolute; z-index:1000; width:max-content; max-width:340px; margin-top:4px;
  padding:8px 10px; border-radius:8px; font-size:12px; font-weight:400; line-height:1.45; white-space:normal;
  color:var(--body-text-color); background:var(--background-fill-primary);
  border:1px solid var(--border-color-primary); box-shadow:0 8px 24px rgba(0,0,0,.25); pointer-events:none; }
span[data-testid="block-info"].has-info + div p,
.block > label:has(> input[data-testid="checkbox"]) + div p {margin:0 !important;}
span[data-testid="block-info"].has-info:hover + div,
.block > label:has(> input[data-testid="checkbox"]):hover + div {display:block;}
.adui-q {position:relative;}
.adui-label:hover .adui-q::after, .adui-q:focus::after {content:attr(data-tip); display:block; top:14px; left:-8px;}
.adui-label {font-size:var(--block-title-text-size); font-weight:var(--block-title-text-weight);
  color:var(--block-title-text-color); padding:8px 10px 4px;}
/* tooltips must be able to extend past their group, or ones on a group's last row get clipped */
.gradio-container .block, .gradio-container .form,
.gradio-container .gr-group, .gradio-container .gr-group > .styler {overflow:visible !important;}
.adui-global {text-align:right; font-size:.85em; padding:6px 12px; border-radius:999px; margin-left:auto;
  width:fit-content; max-width:100%; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.adui-global.idle {opacity:.5;}
.adui-global.busy {background:color-mix(in srgb, var(--color-accent) 18%, transparent); font-weight:500;}
footer {display:none !important;}
"""


# The progress timers poll every 1-2 s, and while the server is down (its terminal was closed or
# restarted) every failed poll adds its own "Connection errored out" toast. Show the first one only,
# and collapse any other identical toasts into one.
JS = """
() => {
  let lastConnError = 0;
  new MutationObserver(() => {
    const items = document.querySelectorAll(".toast-wrap > *");
    if (!items.length) return;
    const now = Date.now(), visible = new Set();
    for (const el of items) {
      const key = el.textContent.trim();
      if (!el.dataset.aduiSeen) {
        el.dataset.aduiSeen = "1";
        if (key.includes("Connection errored out")) {
          if (now - lastConnError < 60000) el.style.display = "none";
          lastConnError = now;
        }
      }
      if (el.style.display === "none") continue;
      if (visible.has(key)) el.style.display = "none"; else visible.add(key);
    }
  }).observe(document.body, {childList: true, subtree: true});
}
"""
