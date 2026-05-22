#!/usr/bin/env bash
# Scenario: subagent forced (investigator dispatch) - 2 rollout steps.
# Expected: each sample [subagent x 1, final x 1] approx 50/50.
# See SPEC §8.2 + §7.2 acceptance gates.

set -euo pipefail
source "$(dirname "$0")/run_qwen36_base.sh"

# Register a single "investigator" subagent the model should dispatch BEFORE
# editing any file. Forces at least one Task/Agent tool_use per sample.
INVESTIGATOR='{"investigator":{"description":"Use FIRST before editing","prompt":"Read PROBLEM_STATEMENT.md, find relevant source files, then return a 3-line summary."}}'
export SWE_CLAUDE_EXTRA_ARGS="--agents '${INVESTIGATOR}' --disable-slash-commands"
export SWE_CC_PROMPT="Before editing any file, dispatch the investigator subagent with the problem statement. Then make your edits based on its summary."

# Disable autoCompact for this scenario to isolate sub-agent segments.
cat > /tmp/cc_sub_settings.json <<'JSON'
{"autoCompactWindow": 1000000}
JSON
export SWE_CLAUDE_EXTRA_ARGS="${SWE_CLAUDE_EXTRA_ARGS} --settings /tmp/cc_sub_settings.json"

export NUM_STEPS_PER_ROLLOUT=2
exec_train "$@"
