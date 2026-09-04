# AlphaFold 3 Server Mode

Server mode lets a lightweight client machine — one without a GPU, the genetic
databases, or the model parameters — run AlphaFold 3 by submitting jobs over
HTTP to a server that runs inside the AlphaFold 3 container. The client uses
the same `run_alphafold.py` entry point and (almost) the same flags as a local
run, and the results are written to the client's `--output_dir` with exactly
the same on-disk layout as a local run.

This is useful when a single GPU host serves many users, or when the databases
and model parameters should exist only on one managed machine.

## Architecture

*   **Client**: any machine with this repository checked out. Reads the fold
    input JSON(s) locally, forwards them together with the flags the user
    explicitly set to the server, streams progress, and downloads and extracts
    the results. The only extra Python dependency on the client is `requests`,
    imported lazily — it is only needed when `--server_mode` is used.
*   **Server**: runs inside the AlphaFold 3 container image (which already
    contains the `server` dependency group and the HMMER binaries). Executes
    one job at a time, each in a fresh spawned Python process that calls the
    usual `run_alphafold.py` entry point — the inference path itself is
    completely standard.

## Server setup

### Prerequisites

1.  The AlphaFold 3 container image built from this repository (see the
    [installation instructions](installation.md)). No Dockerfile changes are
    needed: the image build runs `uv sync --all-groups`, which includes the
    `server` dependency group.
2.  Model parameters (see
    [obtaining model parameters](installation.md#obtaining-model-parameters)).
3.  Genetic databases (see `fetch_databases.sh`).
4.  A GPU, if the server runs inference (the usual case).

### Configuration

The server needs to know where the databases and model parameters live on
*its* filesystem, so that client users never have to. Configuration is read
from two sources, with the config file taking precedence over environment
variables:

*   **Config file** (recommended): a flat JSON object mapping
    `run_alphafold.py` flag names to values, read from
    `$AF3_SERVER_CONFIG` (default `/etc/af3_server/config.json`). Keys starting
    with `_` are ignored (a convention for comments). Entries with `null`
    values are dropped, leaving AlphaFold 3's own default in effect.

    ```json
    {
      "_comment": "Paths as seen inside the container",
      "db_dir": "/data/databases",
      "model_dir": "/data/models",
      "jax_compilation_cache_dir": "/data/jax_cache"
    }
    ```

*   **Environment variables**: `AF3_DB_DIR`, `AF3_MODEL_DIR`,
    `AF3_JACKHMMER_BINARY_PATH`, `AF3_NHMMER_BINARY_PATH`,
    `AF3_HMMALIGN_BINARY_PATH`, `AF3_HMMSEARCH_BINARY_PATH`,
    `AF3_HMMBUILD_BINARY_PATH`, `AF3_JAX_COMPILATION_CACHE_DIR`.

Two more environment variables control the server itself:

*   `AF3_SERVER_DIR` (default `/data/af3_server`): where job state, logs, and
    results are kept. Must be on a writable disk with enough space for
    AlphaFold 3 outputs.
*   `AF3_RUN_ALPHAFOLD_PY` (default `/app/alphafold/run_alphafold.py`): path to
    `run_alphafold.py` inside the container.

### Running the server

```bash
uvicorn af3_server:app --host 0.0.0.0 --port 8000
```

The server processes jobs sequentially, one AlphaFold 3 run at a time; further
submissions wait in an in-memory queue. Note that the queue is not persistent:
if the server process is restarted, queued (not yet running) jobs are lost and
their clients will see the job as failed.

## Client usage

On the client machine, install the repository with the `server` dependency
group — compared to a plain `uv sync` this adds only `requests` to the
environment:

```bash
uv sync --group server
```

Then add `--server_mode=true` and `--server_url` to an otherwise normal
`run_alphafold.py` invocation:

```bash
python run_alphafold.py \
    --server_mode=true \
    --server_url=http://af3-server:8000 \
    --json_path=fold_input.json \
    --output_dir=./af3_output
```

`--json_path` / `--input_dir` and `--output_dir` keep their usual meaning on
the client: where to read inputs from, and where to write the extracted
results. As with a local run, `--force_output_dir=true` is needed to write into
a non-empty output directory.

### Which flags are forwarded

Only flags explicitly passed on the command line are forwarded to the server,
so client-side defaults never clobber the server's configuration. Flags whose
meaning is tied to the client's filesystem or hardware are *not* forwarded;
the server substitutes its own configured values instead:

*   `--json_path`, `--input_dir`, `--output_dir` (handled specially: inputs are
    uploaded, results are downloaded)
*   `--db_dir`, `--model_dir`, all `*_database_path` flags
*   `--jackhmmer_binary_path`, `--nhmmer_binary_path`, `--hmmalign_binary_path`,
    `--hmmsearch_binary_path`, `--hmmbuild_binary_path`
*   `--jax_compilation_cache_dir`, `--gpu_device`

Everything else — e.g. `--num_seeds`, `--num_recycles`, `--num_diffusion_samples`,
`--max_template_date`, `--run_data_pipeline`, `--run_inference`,
`--buckets`, `--flash_attention_implementation` — is forwarded verbatim when
set explicitly.

## Outputs

After a successful run, `--output_dir` contains exactly what a local run would
produce: one subdirectory per fold input (named by its sanitised job name)
with the model input JSON, per-seed/per-sample mmCIF and confidence files,
ranking scores CSV, and the terms of use files. See
[AlphaFold 3 Output](output.md) for the full description.

## HTTP API

The server exposes the following endpoints:

| Endpoint | Description |
|---|---|
| `POST /run` | Submit a job. Body: `{"job_name": str, "fold_inputs": [...], "flags": {...}}`. Returns `{"job_id", "status"}`. |
| `GET /status/{job_id}` | Poll the job status (`queued` / `running` / `done` / `failed`). |
| `GET /progress/{job_id}` | Stream the job's log as it is written. |
| `GET /log/{job_id}` | Download the complete log of a finished job. |
| `GET /result/{job_id}` | List the files produced by the job. |
| `GET /download/{job_id}` | Download the results as a zip archive. |

`run_alphafold.py --server_mode=true` implements the full client side of this
protocol; the endpoints are documented here for anyone writing their own
client or monitoring tooling.

## Troubleshooting

*   **Job fails immediately**: check the server configuration — a missing or
    malformed `db_dir`/`model_dir` is the most common cause. The client prints
    the tail of the server-side log when a job fails; the full log is at
    `GET /log/{job_id}` or `$AF3_SERVER_DIR/runs/{job_id}/run.log` on the
    server.
*   **Client connection errors**: the client retries failed requests five
    times with exponential backoff before giving up. Persistent `5xx` errors
    usually mean the server is overloaded or crashed mid-job; check the server
    process logs.
*   **Slow first job**: the first inference job pays the usual JAX compilation
    cost. Setting `jax_compilation_cache_dir` in the server config makes
    subsequent jobs start faster.
