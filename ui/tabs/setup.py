"""Setup & Models tab: health checks, model inventory, job queue, and maintenance tools."""
from __future__ import annotations

import shutil
import time
from types import SimpleNamespace

import gradio as gr

from core.jobs import ACTIVE, MANAGER, Job, cli_argv
from core.library import LIBRARY
from core.paths import DATA_DIR, UI_INPUT_DIR, cli_python, from_data_rel
from core.system import features, gpu_info, open_folder, refresh_cli_env

FOLDERS = [
    ("Checkpoints", DATA_DIR / "models" / "sd"),
    ("Motion modules & camera LoRAs", DATA_DIR / "models" / "motion-module"),
    ("LoRAs", DATA_DIR / "models" / "lora"),
    ("VAEs", DATA_DIR / "models" / "VAE"),
    ("Embeddings", DATA_DIR / "embeddings"),
]

DOWNLOADS = """
| Model | What it's for | Get it | Put it in |
|---|---|---|---|
| **mm_sd_v15_v2** | Reliable all-round motion module | [huggingface.co/guoyww/animatediff](https://huggingface.co/guoyww/animatediff/tree/main) | `data/models/motion-module` |
| **v3_sd15_mm** + **v3_sd15_adapter** | AnimateDiff v3 (adapter is optional, use it as a camera LoRA) | [huggingface.co/guoyww/animatediff](https://huggingface.co/guoyww/animatediff/tree/main) | `data/models/motion-module` |
| **AnimateDiff-Lightning** (`*_comfyui.safetensors`) | 2–8 step generation (very fast) | [huggingface.co/ByteDance/AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning/tree/main) | `data/models/motion-module` |
| **AnimateLCM** (`AnimateLCM_sd15_t2v.ckpt` + `…_lora.safetensors`) | 4–8 step generation | [huggingface.co/wangfuyun/AnimateLCM](https://huggingface.co/wangfuyun/AnimateLCM/tree/main) | module → `motion-module`, LoRA → `data/models/lora` |
| **v2 camera LoRAs** (`v2_lora_ZoomIn.ckpt`, …) | Pan / zoom / roll | [huggingface.co/guoyww/animatediff](https://huggingface.co/guoyww/animatediff/tree/main) | `data/models/motion-module` |
| SD1.5 checkpoints & LoRAs | The look of your video | [civitai.com](https://civitai.com) | `data/models/sd`, `data/models/lora` |

ControlNet, IP-Adapter, LCM-LoRA, DWPose and the SD1.5 base model download automatically on first use.
Use **original / ComfyUI** releases of motion modules — diffusers-format "MotionAdapter" folders are not supported.
"""


def status_md() -> str:
    env = refresh_cli_env()
    lines = ["### System"]
    gpus = gpu_info()
    if gpus:
        for g in gpus:
            lines.append(f"- ✅ **GPU:** {g['name']} — {g['total_mb'] / 1024:.0f} GB VRAM "
                         f"({g['used_mb'] / 1024:.1f} GB in use), driver {g['driver']}")
    else:
        lines.append("- ❌ **GPU:** no NVIDIA GPU detected (nvidia-smi not found)")
    if env.get("ok"):
        cuda = "✅" if env.get("cuda") else "❌"
        lines.append(f"- {cuda} **PyTorch** {env.get('torch_version')} (CUDA {env.get('cuda_version')}) · "
                     f"diffusers {env.get('diffusers_version')} · animatediff {env.get('animatediff_version')}")
        lines.append(f"- Main environment: `{cli_python()}`")
    else:
        lines.append(f"- ❌ **animatediff environment problem:** {env.get('error')}")
    lines.append("\n### Optional features")
    lines.append("| Feature | Status | How to enable |\n|---|---|---|")
    for name, (ok, how) in features().items():
        lines.append(f"| {name} | {'✅ ready' if ok else '➖ not installed'} | {'' if ok else f'`{how}`'} |")
    return "\n".join(lines)


def inventory_md() -> str:
    LIBRARY.scan()
    kinds = [("checkpoint", "Checkpoints"), ("motion_module", "Motion modules"), ("motion_lora", "Camera LoRAs / adapters"),
             ("lora", "LoRAs"), ("lycoris", "LyCORIS (unsupported)"), ("vae", "VAEs"), ("embedding", "Embeddings"),
             ("unknown", "Unrecognized files")]
    rows = ["| Type | Count | Files |", "|---|---|---|"]
    for k, label in kinds:
        files = LIBRARY.of_kind(k)
        if not files and k in ("lycoris", "unknown"):
            continue
        names = ", ".join(m.label for m in files[:12]) + (" …" if len(files) > 12 else "")
        rows.append(f"| {label} | {len(files)} | {names or '—'} |")
    return "\n".join(rows)


def jobs_table():
    rows = []
    for j in reversed(MANAGER.jobs()):
        rows.append([time.strftime("%H:%M:%S", time.localtime(j.created)), j.title, j.status,
                     f"{int(j.elapsed // 60)}:{int(j.elapsed % 60):02d}", j.error or "", j.id])
    return rows


def build(demo: gr.Blocks) -> SimpleNamespace:
    status = gr.Markdown("Checking…")
    recheck = gr.Button("Re-check", size="sm")
    gr.Markdown("### Your models")
    with gr.Row():
        for label, path in FOLDERS:
            gr.Button(label, size="sm").click(lambda p=path: open_folder(p), None, None, queue=False)
    inventory = gr.Markdown()
    rescan = gr.Button("Rescan models", size="sm")
    with gr.Accordion("Recommended downloads", open=False):
        gr.Markdown(DOWNLOADS)

    with gr.Accordion("Job queue", open=True):
        jobs = gr.Dataframe(headers=["Queued", "Job", "Status", "Time", "Problem", "id"],
                            datatype=["str"] * 6, interactive=False, wrap=True, value=jobs_table)
        with gr.Row():
            cancel_all = gr.Button("Cancel all", variant="stop", size="sm")
            clear_done = gr.Button("Clear finished", size="sm")
            view_log = gr.Dropdown(label="Show log of job", choices=[], scale=3)
        log = gr.Textbox(label="Log", lines=14, max_lines=14, interactive=False, elem_classes=["adui-log"])

    with gr.Accordion("Tools", open=False):
        gr.Markdown("**Fix checkpoint** — for models that fail with `'Attention' object has no attribute 'to_to_k'`.")
        with gr.Row():
            fix_ckpt = gr.Dropdown(LIBRARY.choices("checkpoint"), label="Checkpoint", scale=4)
            fix_btn = gr.Button("Fix it", scale=1)
        fix_msg = gr.Markdown()
        gr.Markdown("**Clean up** uploaded inputs (driving videos, reference images) older than a week. "
                    "Past runs that used them can still be viewed but not re-run as-is.")
        with gr.Row():
            clean_btn = gr.Button("Clean up old inputs", size="sm")
            clean_msg = gr.Markdown()

    demo.load(lambda: (status_md(), inventory_md()), None, [status, inventory])
    recheck.click(status_md, None, [status])
    rescan.click(inventory_md, None, [inventory])

    def refresh_jobs():
        jl = MANAGER.jobs()
        return jobs_table(), gr.update(choices=[(f"{j.title} [{j.status}]", j.id) for j in reversed(jl)])

    timer = gr.Timer(3.0)
    timer.tick(refresh_jobs, None, [jobs, view_log], show_progress="hidden", queue=False)

    def show_log(jid):
        j = MANAGER.get(jid)
        if not j:
            return ""
        if j.log_path and j.log_path.is_file():
            return j.log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
        return j.tail(400)

    view_log.change(show_log, [view_log], [log])

    def do_cancel_all():
        n = sum(MANAGER.cancel(j.id) for j in MANAGER.jobs() if j.status in ACTIVE)
        gr.Info(f"Cancelled {n} job(s).")
        return jobs_table()

    cancel_all.click(do_cancel_all, None, [jobs])
    clear_done.click(lambda: (MANAGER.clear_finished(), jobs_table())[1], None, [jobs])

    def do_fix(rel):
        if not rel:
            raise gr.Error("Pick a checkpoint.")
        job = MANAGER.submit(Job(kind="tool", title=f"Fix checkpoint · {rel}",
                                 argv=cli_argv("fix-checkpoint", from_data_rel(rel))))
        return f"Queued — watch it in the job queue above (job {job.id})."

    fix_btn.click(do_fix, [fix_ckpt], [fix_msg])

    def do_clean():
        if not UI_INPUT_DIR.is_dir():
            return "Nothing to clean."
        cutoff = time.time() - 7 * 86400
        removed = 0
        for d in UI_INPUT_DIR.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return f"Removed {removed} old input folder(s)."

    clean_btn.click(do_clean, None, [clean_msg])
    return SimpleNamespace()
