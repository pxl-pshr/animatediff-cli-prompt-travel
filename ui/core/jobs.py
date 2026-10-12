"""Background job queue that runs the animatediff CLI in a subprocess.

One GPU job runs at a time; the rest wait in a FIFO queue. Jobs are owned by
the server, not by a browser tab, so closing or refreshing the page never
kills a render, and any tab can watch progress or cancel.
"""
from __future__ import annotations

import codecs
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .errors import explain_failure
from .paths import REPO_ROOT, UI_LOG_DIR, cli_python

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# tqdm.rich (what the CLI uses), redrawn with \r when FORCE_COLOR=1:
#   "Preprocessing images (x)  17% ----- ---- 2/12  [ 0:00:00 < 0:00:02 , 7 it/s ]"
# plain tqdm (HF downloads):
#   " 45%|####5     | 9/20 [00:05<00:06,  1.80it/s]"
BAR_RE = re.compile(
    r"(?:^|\s)(\d{1,3})%\s.*?\s([\d,]+)/([\d,]+)\s+\[\s*([\d:]+)\s*<\s*([\d:?-]+)"
    r"|(\d{1,3})%\|[^|]*\|\s*([\d,.]+\w?)/([\d,.]+\w?)\s*\[([\d:]+)<([\d:?]+)")
# End of a rich bar; rich appends the next log line right after it with no newline.
BAR_TAIL_RE = re.compile(r"\[\s*[\d:]+\s*<\s*[\d:?-]+\s*,[^\]]*\]")
SRC_SUFFIX_RE = re.compile(r"\s{2,}[\w.]+\.py:\d+\s*$")     # rich's right-aligned "cli.py:313"
SAVE_DIR_RE = re.compile(r"Will save outputs to \.?[\\/]?(.+?)\s*$")

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
ACTIVE = {QUEUED, RUNNING}

# Log line fragments -> human readable stage names.
STAGES = [
    ("Using generation config", "Reading config"),
    ("Using base model", "Checking base model"),
    ("Downloading", "Downloading models (first run only)"),
    ("Will save outputs", "Preparing"),
    ("preprocess", "Preprocessing ControlNet / IP-Adapter inputs"),
    ("Loading", "Loading models"),
    ("Sending pipeline to device", "Moving model to GPU"),
    ("Initialization complete", "Starting generation"),
    ("Running generation", "Generating"),
    ("Running RIFE", "Interpolating frames (RIFE)"),
    ("Generation complete, saving", "Saving frames"),
    ("Creating ffmpeg encoder", "Encoding video"),
    ("Generation complete", "Finishing"),
    ("Done, exiting", "Done"),
]


@dataclass
class Job:
    kind: str
    title: str
    argv: list[str]
    out_root: Path | None = None              # where the CLI will create its run folder
    on_success: Callable[["Job"], None] | None = None
    meta: dict = field(default_factory=dict)

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    stage: str = "Waiting in queue"
    step: int = 0
    total: int = 0
    eta: str = ""
    save_dir: Path | None = None
    error: str | None = None
    hint: str | None = None
    lines: deque = field(default_factory=lambda: deque(maxlen=400))
    log_path: Path | None = None
    _proc: subprocess.Popen | None = None
    _cancel: bool = False

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    @property
    def fraction(self) -> float:
        return self.step / self.total if self.total else 0.0

    def status_line(self) -> str:
        if self.status == QUEUED:
            return "Waiting in queue"
        if self.status == RUNNING:
            mins, secs = divmod(int(self.elapsed), 60)
            prog = f" — step {self.step}/{self.total}" if self.total else ""
            eta = f", ~{self.eta} left in this pass" if self.eta and self.total else ""
            return f"{self.stage}{prog}{eta} ({mins}:{secs:02d} elapsed)"
        if self.status == DONE:
            return f"Finished in {int(self.elapsed // 60)}:{int(self.elapsed % 60):02d}"
        if self.status == CANCELLED:
            return "Cancelled"
        return f"Failed: {self.error or 'see log'}"

    def tail(self, n: int = 200) -> str:
        return "\n".join(list(self.lines)[-n:])


class JobManager:
    def __init__(self) -> None:
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        UI_LOG_DIR.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, name="animatediff-jobs", daemon=True).start()

    # ---- queue API -------------------------------------------------------
    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs.append(job)
        self._wake.set()
        return job

    def get(self, job_id: str | None) -> Job | None:
        if not job_id:
            return None
        with self._lock:
            return next((j for j in self._jobs if j.id == job_id), None)

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs)

    def current(self) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs if j.status == RUNNING), None)

    def latest(self, kind: str | None = None) -> Job | None:
        with self._lock:
            pool = [j for j in self._jobs if kind is None or j.kind == kind]
        return pool[-1] if pool else None

    def queue_position(self, job: Job) -> int:
        with self._lock:
            queued = [j for j in self._jobs if j.status == QUEUED]
        return queued.index(job) + 1 if job in queued else 0

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status not in ACTIVE:
            return False
        job._cancel = True
        if job.status == QUEUED:
            job.status = CANCELLED
            job.finished = time.time()
            return True
        if job._proc and job._proc.poll() is None:
            _kill_tree(job._proc)
        return True

    def clear_finished(self) -> None:
        with self._lock:
            self._jobs = [j for j in self._jobs if j.status in ACTIVE]

    # ---- worker ----------------------------------------------------------
    def _next(self) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs if j.status == QUEUED), None)

    def _worker(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            try:
                self._run(job)
            except Exception as e:  # never let the worker thread die
                job.status = FAILED
                job.error = f"UI error while running job: {e!r}"
                job.finished = time.time()

    def _run(self, job: Job) -> None:
        job.status = RUNNING
        job.started = time.time()
        job.stage = "Starting"
        job.log_path = UI_LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{job.kind}-{job.id}.log"
        before = _list_dirs(job.out_root)

        env = os.environ.copy()
        env.update({
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "COLUMNS": "300",            # stop rich from wrapping log lines
            "FORCE_COLOR": "1",          # rich only redraws progress bars live on a "terminal"
            "_TYPER_STANDARD_TRACEBACK": "1",
            # see src/animatediff/__init__.py: avoids a Windows hang in old huggingface_hub downloads
            "HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD": str(1 << 62),
        })
        env.pop("VIRTUAL_ENV", None)

        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True

        with open(job.log_path, "w", encoding="utf-8") as log:
            log.write("$ " + " ".join(f'"{a}"' if " " in a else a for a in job.argv) + "\n\n")
            try:
                proc = subprocess.Popen(
                    job.argv, cwd=str(REPO_ROOT), env=env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    **kwargs,
                )
            except OSError as e:
                job.status = FAILED
                job.error = f"Could not start the animatediff CLI: {e}"
                job.hint = f"Expected the main venv interpreter at {cli_python()}. Run the install steps from the README first."
                job.finished = time.time()
                return
            job._proc = proc
            self._pump(job, proc, log)
            job.returncode = proc.wait()

        job.finished = time.time()
        job._proc = None

        if job.save_dir is None:
            job.save_dir = _newest_new_dir(job.out_root, before)

        if job._cancel:
            job.status = CANCELLED
            job.stage = "Cancelled"
            return
        if job.returncode == 0 and not _looks_failed(job):
            job.status = DONE
            job.stage = "Done"
            if job.on_success:
                try:
                    job.on_success(job)
                except Exception as e:
                    job.lines.append(f"[ui] post-processing failed: {e!r}")
            return
        job.status = FAILED
        job.error, job.hint = explain_failure(list(job.lines), job.returncode)

    def _pump(self, job: Job, proc: subprocess.Popen, log) -> None:
        """Read stdout in raw chunks so tqdm's carriage-return updates arrive live."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf = ""
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf += decoder.decode(chunk)
            # Split on \r\n / \n (real lines) and bare \r (tqdm redraws).
            while True:
                m = re.search(r"\r\n|\n|\r", buf)
                if not m:
                    break
                if m.group() == "\r" and m.end() == len(buf):
                    break  # might be the first half of \r\n; wait for more data
                piece, buf = buf[: m.start()], buf[m.end():]
                self._split(job, ANSI_RE.sub("", piece), m.group() == "\r", log)
        if buf.strip():
            self._split(job, ANSI_RE.sub("", buf), False, log)

    def _split(self, job: Job, text: str, is_redraw: bool, log) -> None:
        tail = BAR_TAIL_RE.search(text)
        if tail and text[tail.end():].strip():
            self._handle(job, text[: tail.end()], True, log)
            self._split(job, text[tail.end():], is_redraw, log)
        else:
            self._handle(job, text, is_redraw, log)

    def _handle(self, job: Job, text: str, is_redraw: bool, log) -> None:
        line = text.rstrip()
        if not line.strip():
            return
        m = BAR_RE.search(line)
        if m:
            g = m.groups()
            pct, n, total, eta = (g[0], g[1], g[2], g[4]) if g[0] is not None else (g[5], g[6], g[7], g[9])
            try:
                job.step, job.total = int(n.replace(",", "")), int(total.replace(",", ""))
            except ValueError:   # download bars report sizes like 1.2G/3.4G
                job.step, job.total = int(pct), 100
            job.eta = eta if eta and "?" not in eta and "-" not in eta else ""
            desc = line[: m.start()].strip()
            if desc and not desc[0].isdigit():
                job.stage = desc
            elif job.stage.startswith(("Preprocessing", "Saving Preprocessed", "Starting generation")):
                job.stage = "Generating (denoising)"
            if job.step >= job.total and job.total:   # keep finished bars only
                short = re.sub(r"[-━╸╺ ]{6,}", " ", line).strip()
                job.lines.append(short)
                log.write(short + "\n")
            return
        if is_redraw:
            return
        line = SRC_SUFFIX_RE.sub("", line)
        job.lines.append(line)
        log.write(line + "\n")
        log.flush()
        for needle, stage in STAGES:
            if needle in line:
                job.stage = stage
                if stage != "Generating":
                    job.step = job.total = 0
                break
        sm = SAVE_DIR_RE.search(line)
        if sm and job.save_dir is None:
            cand = (REPO_ROOT / sm.group(1).strip()).resolve()
            if cand.is_dir():
                job.save_dir = cand


def _looks_failed(job: Job) -> bool:
    # typer can exit 0 after printing a traceback in some code paths.
    tail = list(job.lines)[-15:]
    return any(l.startswith("Traceback (most recent call last)") for l in tail)


def _list_dirs(root: Path | None) -> set[str]:
    if not root or not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir()}


def _newest_new_dir(root: Path | None, before: set[str]) -> Path | None:
    if not root or not root.is_dir():
        return None
    new = [p for p in root.iterdir() if p.is_dir() and p.name not in before]
    return max(new, key=lambda p: p.stat().st_mtime) if new else None


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.kill()


def cli_argv(*args: str) -> list[str]:
    return [str(cli_python()), "-m", "animatediff", *[str(a) for a in args]]


MANAGER = JobManager()
