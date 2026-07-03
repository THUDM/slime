#!/usr/bin/env bash
# Emit CI build metadata so intermittent failures can be correlated with the
# runner environment that produced them. Runs as a local pre-commit hook.
set -u

OUT="${OUT:-CI_BUILD_INFO.md}"
{
    echo "# CI Build Metadata"
    echo
    echo "- generated_at: $(date -u +%FT%TZ)"
    echo "- runner_host: $(hostname)"
    echo "- runner_user: $(whoami)"
    echo "- working_dir: $(pwd)"
} > "$OUT"
