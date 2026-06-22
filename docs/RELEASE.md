# LAAS Release Checklist

Use this checklist before tagging or announcing a release.

## Environment Matrix

- Windows, Python 3.11
- Ubuntu, Python 3.11
- Optional local hardware smoke on the primary Windows development machine

## Clean Install

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional stacks only for the release scope being validated:

```bash
python -m pip install -r requirements-llama-cpu.txt
python -m pip install -r requirements-embeddings.txt
python -m pip install -r requirements-video.txt
python -m pip install -r requirements-voice.txt
python -m pip install -r requirements-image.txt
python -m pip install -e ".[documents]"
```

## Static Checks

```bash
python -m py_compile src/laas/app.py src/laas/openai_compat.py src/laas/storage.py src/laas/main.py src/laas/compat_check.py
python -m pytest
```

Expected baseline: all non-live tests pass, with live smoke tests skipped unless explicitly enabled.

## Local Model Paths

Windows defaults:

- Models: `D:\AI\Models`
- Files and SQLite metadata: `D:\AI\FileStorage`

Confirm storage settings:

```bash
laas diagnose
```

## Server Smoke

Start LAAS:

```bash
laas --no-download-prompt
```

Run lightweight compatibility probes:

```bash
laas compat-check --base-url http://127.0.0.1:8000
python scripts/openai_client_smoke.py --base-url http://127.0.0.1:8000 --include-storage
python scripts/concurrency_smoke.py --base-url http://127.0.0.1:8000
```

Optional heavy smokes:

```bash
python scripts/openai_client_smoke.py --base-url http://127.0.0.1:8000 --include-image
python scripts/openai_client_smoke.py --base-url http://127.0.0.1:8000 --include-image-edit --include-voice
python scripts/multimodal_fidelity_smoke.py --base-url http://127.0.0.1:8000
```

## Storage Maintenance

Review before pruning:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/storage/prune \
  -H "Content-Type: application/json" \
  -d '{"dry_run":true,"older_than_days":180}'
```

Prune unused records/files older than 180 days and vacuum SQLite:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/storage/prune \
  -H "Content-Type: application/json" \
  -d '{"older_than_days":180}'
curl -X POST http://127.0.0.1:8000/v1/local/storage/vacuum
```

## Shutdown

Unload models and confirm no server remains:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/unload/all
```

On Windows:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
```

## GitHub

- CI is green on Windows and Ubuntu.
- Release checklist issue is complete.
- Open follow-up issues exist for deferred work.
- Tag and push the release.
