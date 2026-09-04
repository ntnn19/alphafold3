"""AlphaFold 3 in-container HTTP server.

Runs *inside* the same Singularity container as AlphaFold 3 itself. Accepts
job submissions via HTTP, launches each job in a fresh Python child process
(``multiprocessing`` spawn context) that imports ``run_alphafold`` and calls
its ``main`` via ``absl.app.run``. No nested ``singularity exec``.

Wire protocol:

    POST /run
        body: {"job_name": str,
               "fold_inputs": [ <raw AF3 fold-input JSON object>, ... ],
               "flags":       { <forwarded run_alphafold.py flags> }}
        -> {"job_id": "<uuid>", "status": "queued"}

    GET /status/{id}       cheap poll
    GET /progress/{id}     text stream tailing run.log
    GET /log/{id}          full run.log
    GET /result/{id}       manifest of run_dir
    GET /download/{id}     zip of run_dir

The server has no hard-coded model / DB / extra-flag configuration: every
knob is passed from the client, matching exactly what the user typed on
their local ``run_alphafold.py`` CLI.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import resource
import shutil
import sys
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────
SERVER_DIR = Path(os.environ.get("AF3_SERVER_DIR", "/data/af3_server"))
RUN_DIR = SERVER_DIR / "runs"
STATE_DIR = SERVER_DIR / "state"

for _d in (RUN_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Path to run_alphafold.py inside the container. Overridable so the same
# server code works if the container layout changes.
RUN_ALPHAFOLD_PY = os.environ.get(
    "AF3_RUN_ALPHAFOLD_PY", "/app/alphafold/run_alphafold.py"
)

# Server-side defaults for the AF3 CLI flags that hold paths/values visible
# only inside the server's filesystem. The client MUST NOT need to know
# these: in --server_mode the client drops all local-only path flags before
# forwarding, so this dict is what makes the child invocation actually
# find its databases and weights.
#
# Precedence (see _build_argv):
#   1. --input_dir / --output_dir set by server (always).
#   2. Client-forwarded flags (rare in practice — the client drops these).
#   3. Server config file at $AF3_SERVER_CONFIG (default:
#      /etc/af3_server/config.json).
#   4. Env-var shortcuts: AF3_DB_DIR, AF3_MODEL_DIR, AF3_*_BINARY_PATH,
#      AF3_JAX_COMPILATION_CACHE_DIR. Config file overrides env vars if
#      both are set.
#
# The config file is a JSON object mapping AF3 flag names -> values. It is
# the operator's single source of truth for the whole server-side
# invocation. Any None / missing entry means "let AF3's own default handle
# that flag" — which is only OK for flags whose upstream default actually
# works on this host.

_SERVER_CONFIG_PATH = Path(
    os.environ.get("AF3_SERVER_CONFIG", "/etc/af3_server/config.json")
)


def _load_server_defaults() -> dict[str, Any]:
    """Build the server's default AF3-flag dict.

    Sources (later ones override earlier):
      1. Env-var shortcuts for the most common flags.
      2. JSON config file at ``$AF3_SERVER_CONFIG`` (a flat dict of
         ``flag_name -> value``).

    Any entry whose value is ``None`` is dropped: the child inherits
    AF3's own absl default for that flag. Any entry whose value is a
    non-empty string, bool, int, or float is forwarded verbatim on the
    child argv.
    """
    defaults: dict[str, Any] = {
        "db_dir": os.environ.get("AF3_DB_DIR"),
        "model_dir": os.environ.get("AF3_MODEL_DIR"),
        "jackhmmer_binary_path": os.environ.get("AF3_JACKHMMER_BINARY_PATH"),
        "nhmmer_binary_path": os.environ.get("AF3_NHMMER_BINARY_PATH"),
        "hmmalign_binary_path": os.environ.get("AF3_HMMALIGN_BINARY_PATH"),
        "hmmsearch_binary_path": os.environ.get("AF3_HMMSEARCH_BINARY_PATH"),
        "hmmbuild_binary_path": os.environ.get("AF3_HMMBUILD_BINARY_PATH"),
        "jax_compilation_cache_dir": os.environ.get(
            "AF3_JAX_COMPILATION_CACHE_DIR"
        ),
    }

    if _SERVER_CONFIG_PATH.is_file():
        try:
            loaded = json.loads(_SERVER_CONFIG_PATH.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed AF3 server config at {_SERVER_CONFIG_PATH}: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"AF3 server config at {_SERVER_CONFIG_PATH} must be a JSON"
                f" object mapping flag names -> values."
            )
        # Config file wins over env vars. Keys starting with '_' are
        # comment-only and ignored (convention for JSON without comments).
        for k, v in loaded.items():
            if k.startswith("_"):
                continue
            defaults[k] = v

    # Drop None entries so we never emit '--flag=None' on the child argv.
    return {k: v for k, v in defaults.items() if v is not None}


_SERVER_DEFAULT_FLAGS: dict[str, Any] = _load_server_defaults()

app = FastAPI(title="AlphaFold 3 in-container server")


# ── Models ────────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    job_name: str
    status: JobStatus
    submitted_at: str
    run_dir: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class RunRequest(BaseModel):
    job_name: str
    fold_inputs: list[Any]  # list of raw AF3 fold-input JSON objects
    flags: dict[str, Any] = {}


# ── State ────────────────────────────────────────────────
_queue: asyncio.Queue[str] = asyncio.Queue()
_jobs: dict[str, JobInfo] = {}


# ── Helpers ────────────────────────────────────────────────
def _raise_fd_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(hard, 65535)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))


def _fmt_flag_value(v: Any) -> str:
    """Format a Python value for an absl-style ``--flag=value`` argument."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        # ``flags.DEFINE_list`` accepts comma-joined strings.
        return ",".join(str(x) for x in v)
    return str(v)


def _build_argv(inputs_dir: Path, run_dir: Path, flags: dict[str, Any]) -> list[str]:
    """Build the argv passed to ``absl.app.run(main, argv=argv)`` in the child.

    Precedence:
      1. ``--input_dir`` / ``--output_dir`` — set by the server; any values
         the client tried to forward for those keys are ignored.
      2. Client-forwarded flags win over server defaults (the client already
         drops local-only path flags, so this only overrides in the rare
         case where the client explicitly re-added one).
      3. Server-side defaults (``_SERVER_DEFAULT_FLAGS``) fill in the
         database/model paths and HMMER binaries so the user never has to
         know them in ``--server_mode``.
    """
    client_flags = {
        k: v for k, v in flags.items()
        if k not in ("input_dir", "output_dir", "json_path")
    }

    argv: list[str] = [
        RUN_ALPHAFOLD_PY,
        f"--input_dir={inputs_dir}",
        f"--output_dir={run_dir}",
    ]

    # Merge: server defaults + client-forwarded, client wins on collision.
    effective: dict[str, Any] = {}
    for k, v in _SERVER_DEFAULT_FLAGS.items():
        if v is not None:
            effective[k] = v
    effective.update(client_flags)

    for k, v in effective.items():
        if v is None:
            continue
        argv.append(f"--{k}={_fmt_flag_value(v)}")
    return argv


def _child_entrypoint(argv: list[str], log_path: str) -> None:
    """Run in a fresh spawned Python interpreter, one per job.

    Redirects stdout/stderr to ``log_path`` (unbuffered), then calls
    ``absl.app.run(run_alphafold.main, argv=argv)``. Exit code is set by
    ``sys.exit`` so the parent can tell success from failure.
    """
    # Redirect early so import errors also land in the log file.
    log_f = open(log_path, "w", buffering=1)  # line-buffered
    os.dup2(log_f.fileno(), sys.stdout.fileno())
    os.dup2(log_f.fileno(), sys.stderr.fileno())

    try:
        # Make sure the directory containing run_alphafold.py is importable.
        script_path = Path(argv[0])
        script_dir = str(script_path.parent)
        if script_dir and script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        from absl import app as absl_app  # noqa: WPS433 (runtime import inside child)

        import run_alphafold  # noqa: WPS433

        absl_app.run(run_alphafold.main, argv=argv)
        sys.exit(0)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        print(f"[af3_server child] FATAL: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)


# ── Worker ────────────────────────────────────────────────
async def _run_job(job: JobInfo, envelope: RunRequest) -> None:
    run_dir = Path(job.run_dir)
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    # Persist each fold input as a JSON file that AF3's ``load_fold_inputs_from_dir``
    # will pick up. We keep the raw JSON the client sent so no re-serialization
    # can subtly change the input.
    for i, fold_input in enumerate(envelope.fold_inputs):
        (inputs_dir / f"input_{i:04d}.json").write_text(json.dumps(fold_input))

    argv = _build_argv(inputs_dir, run_dir, envelope.flags)

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_child_entrypoint, args=(argv, str(log_path)))
    proc.start()

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, proc.join)
    finally:
        job.finished_at = datetime.utcnow().isoformat()

    exit_code = proc.exitcode
    if exit_code == 0:
        job.status = JobStatus.DONE
    else:
        job.status = JobStatus.FAILED
        job.error = f"child exited with code {exit_code}"

    _jobs[job.job_id] = job


async def _worker() -> None:
    while True:
        job_id = await _queue.get()
        try:
            job = _jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow().isoformat()

            # Envelope is stored on disk to survive queue delays without
            # holding it in memory.
            envelope_path = STATE_DIR / job_id / "envelope.json"
            envelope_raw = json.loads(envelope_path.read_text())
            envelope = RunRequest(**envelope_raw)

            await _run_job(job, envelope)
        except Exception as e:  # noqa: BLE001
            job = _jobs.get(job_id)
            if job is not None:
                job.status = JobStatus.FAILED
                job.error = f"worker error: {e}"
                job.finished_at = datetime.utcnow().isoformat()
        finally:
            _queue.task_done()


@app.on_event("startup")
async def _startup() -> None:
    _raise_fd_limit()
    asyncio.create_task(_worker())


# ── Endpoints ────────────────────────────────────────────
@app.post("/run")
async def run(req: RunRequest):
    if not req.fold_inputs:
        raise HTTPException(400, "fold_inputs must contain at least one entry")

    job_id = str(uuid.uuid4())

    state_dir = STATE_DIR / job_id
    state_dir.mkdir(parents=True, exist_ok=True)
    # Version-tolerant serialization (Pydantic v1: .dict(); v2: .model_dump()).
    to_dict = getattr(req, "model_dump", None) or req.dict
    (state_dir / "envelope.json").write_text(json.dumps(to_dict()))

    run_dir = RUN_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    job = JobInfo(
        job_id=job_id,
        job_name=req.job_name,
        status=JobStatus.QUEUED,
        submitted_at=datetime.utcnow().isoformat(),
        run_dir=str(run_dir),
    )
    _jobs[job_id] = job
    await _queue.put(job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    to_dict = getattr(job, "model_dump", None) or job.dict
    return to_dict()


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")

    async def stream():
        log_path = Path(job.run_dir) / "run.log"
        last_size = 0

        # Follow log until the job leaves RUNNING (or never started but failed
        # in a way that never wrote a log file).
        while job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            await asyncio.sleep(0.5)
            if log_path.exists():
                size = log_path.stat().st_size
                if size > last_size:
                    with open(log_path, "r") as f:
                        f.seek(last_size)
                        yield f.read()
                    last_size = size

        # Flush any final bytes written just as the job finished.
        if log_path.exists():
            size = log_path.stat().st_size
            if size > last_size:
                with open(log_path, "r") as f:
                    f.seek(last_size)
                    yield f.read()

        yield f"\n[done] status={job.status} error={job.error}\n"

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/log/{job_id}")
async def log(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    log_path = Path(job.run_dir) / "run.log"
    if not log_path.exists():
        raise HTTPException(404, "No log yet")
    return FileResponse(log_path, media_type="text/plain")


@app.get("/result/{job_id}")
async def result(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    run_dir = Path(job.run_dir)
    if not run_dir.exists():
        raise HTTPException(404, "Run directory missing")

    files = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()]
    return JSONResponse(
        {
            "job_id": job_id,
            "status": job.status,
            "run_dir": str(run_dir),
            "files": files,
        }
    )


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    run_dir = Path(job.run_dir)
    if not run_dir.exists():
        raise HTTPException(404, "Run directory missing")

    # Zip only the AF3-produced content (skip the ``inputs/`` staging dir and
    # the ``run.log`` — the client already has the input and can pull the log
    # separately if it wants it).
    archive_base = STATE_DIR / job_id / "result"
    if archive_base.with_suffix(".zip").exists():
        return FileResponse(archive_base.with_suffix(".zip"), media_type="application/zip")

    # Build the archive from a filtered view: everything except ``inputs/``
    # and ``run.log``.
    staging = STATE_DIR / job_id / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for entry in run_dir.iterdir():
        if entry.name in ("inputs", "run.log"):
            continue
        dst = staging / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)

    zip_path = shutil.make_archive(str(archive_base), "zip", staging)
    shutil.rmtree(staging)
    return FileResponse(zip_path, media_type="application/zip")
