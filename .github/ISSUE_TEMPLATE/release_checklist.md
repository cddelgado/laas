---
name: Release checklist
about: Track a LAAS release validation pass.
title: "Release checklist: vX.Y.Z"
labels: ["release"]
assignees: ""
---

Canonical release process: `docs/RELEASE.md`

## CI

- [ ] Windows / Python 3.11 passed
- [ ] Ubuntu / Python 3.11 passed

## Local Validation

- [ ] Clean install completed
- [ ] `python -m pytest` passed
- [ ] `laas compat-check --base-url http://127.0.0.1:8000` passed
- [ ] `scripts/openai_client_smoke.py --include-storage` passed
- [ ] Optional heavy smoke scope documented

## Storage

- [ ] Storage status reviewed
- [ ] 180-day unused prune dry run reviewed
- [ ] Prune/vacuum performed if appropriate

## Shutdown

- [ ] `/v1/local/unload/all` completed
- [ ] No server is listening on port 8000

## Release

- [ ] Follow-up issues created for deferred work
- [ ] Tag pushed
- [ ] Release notes published
