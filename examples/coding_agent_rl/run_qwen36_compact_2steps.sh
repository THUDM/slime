#!/usr/bin/env bash
# Scenario: autoCompact forced (low window) - 2 rollout steps.
# Expected: each sample [pre_wipe x (1-3), final x 1] approx 60/40.
# See SPEC §8.3 + §7.2 acceptance gates.

set -euo pipefail
source "$(dirname "$0")/run_qwen36_base.sh"

# Disable subagent tools so the only segment-emit path is autoCompact wipe.
export SWE_CLAUDE_EXTRA_ARGS="--disable-slash-commands --disallowedTools Agent Task"

# 100K window = aggressive compact, fires several times during a typical
# debug-and-fix loop on Qwen3.6-35B.
cat > /tmp/cc_compact_settings.json <<'JSON'
{"autoCompactWindow": 100000}
JSON
export SWE_CLAUDE_EXTRA_ARGS="${SWE_CLAUDE_EXTRA_ARGS} --settings /tmp/cc_compact_settings.json"

# Longer wallclock budget so compact has time to fire multiple times.
export SWE_TIME_BUDGET_SEC=1200
export NUM_STEPS_PER_ROLLOUT=2
exec_train "$@"
