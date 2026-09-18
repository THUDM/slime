---
name: release
description: Prepare and verify a slime release, including version bumps, stable Docker patch snapshots, Docker/conda dependency alignment, and release-specific validation. Use when cutting or auditing a slime release.
---

# Release slime

Prepare a release without publishing, tagging, pushing images, or changing the
dependency baseline unless the user explicitly requests those external or
scope-expanding actions.

## Establish the release baseline

- Inspect the worktree and preserve unrelated user changes.
- Compare the previous release tag and release commit to identify the current
  repository conventions.
- Confirm the requested slime version and the current stable SGLang version
  from `docker/Dockerfile` and `docker/README.md`.
- Do not pull in an unmerged SGLang/Docker upgrade merely because a newer
  branch exists. Treat that as a separate decision.

## Update release versions

- Set the package version in `setup.py`.
- Set the documentation version in `docs/conf.py`.
- Bump `docker/version.txt` to a unique image version following its existing
  dated naming convention.
- Search the repository for the old slime version and review every remaining
  occurrence instead of replacing unrelated dependency versions.

## Freeze the stable Docker patches

- Treat `docker/patch/latest/` as the patch stack for the current Docker base.
- Snapshot it exactly into `docker/patch/<stable-sglang-version>/`. At release
  time, the two directories must contain the same patch filenames and bytes.
- Preserve older SGLang patch directories. Remove an obsolete file from the
  current stable snapshot only after confirming it is absent from `latest`.
- Verify the stable patch stack applies in Dockerfile order to clean checkouts
  of the pinned SGLang and Megatron commits. Do not validate against a dirty
  developer checkout.

## Audit Docker and conda together

Compare `build_conda.sh` with `docker/Dockerfile`, `docker/justfile`, and the
stable patch snapshot. Check at least:

- SGLang version, commit, CUDA variant, `sglang-kernel`, and `sgl-deep-gemm`;
- Megatron, torch-memory-saver, FlashQLA, and other shared source pins;
- `PATCH_VERSION`, patch filenames, application order, optional patches, and
  failure-on-conflict behavior;
- PyTorch, torchvision, torchaudio, CUDA Python, Transformer Engine, router,
  NumPy, and SciPy pins;
- whether dependency resolution can undo a compatibility pin later in the
  script; reassert and validate such pins after the resolving install;
- intentional differences such as conda being CUDA-12-only, omitting FA3, or
  not rebuilding feature-specific DeepGEMM/DeepEP forks. Keep a difference
  only when the release CI scope makes it intentional.

Prefer direct loops over duplicated patch-application blocks while preserving
required-versus-optional semantics and useful failure messages.

## Validate before handoff

Run the checks that are available locally:

- `python .claude/skills/release/scripts/check_release.py --repo .
  --expected-version <version>`;
- `python setup.py --version`;
- `bash -n build_conda.sh`;
- byte-for-byte comparison of `docker/patch/latest/` and the stable snapshot;
- patch parsing plus clean-checkout application against the pinned upstream
  commits;
- `git diff --check` and a final review of the complete release diff.

Run the release conda CI and relevant Docker builds when the environment and
requested scope permit. The conda workflow is selected by a PR title containing
`[release]`. Explicitly report any full build or GPU validation that was not run.
