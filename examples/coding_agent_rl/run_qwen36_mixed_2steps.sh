#!/usr/bin/env bash
# Scenario: compact + subagent mixed - 2 rollout steps.
# Expected: each sample [pre_wipe x 1, subagent x 1, final x 1] approx 1/3 each.
# See SPEC §8.4 + §7.2 acceptance gates.

set -euo pipefail
source "$(dirname "$0")/run_qwen36_base.sh"

INVESTIGATOR='{"investigator":{"description":"Use FIRST before editing","prompt":"Read PROBLEM_STATEMENT.md, find relevant source files, then return a 3-line summary."}}'
export SWE_CLAUDE_EXTRA_ARGS="--agents '${INVESTIGATOR}' --disable-slash-commands"
# Prompt that encourages enough context to trigger autoCompact AND a sub dispatch.
export SWE_CC_PROMPT="Before editing, dispatch investigator. After it returns, read every related source file (this will overflow the autoCompact window). Then make your edits."

# 100K window for active compact.
cat > /tmp/cc_mixed_settings.json <<'JSON'
{"autoCompactWindow": 100000}
JSON
export SWE_CLAUDE_EXTRA_ARGS="${SWE_CLAUDE_EXTRA_ARGS} --settings /tmp/cc_mixed_settings.json"

export SWE_TIME_BUDGET_SEC=1500
export NUM_STEPS_PER_ROLLOUT=2
exec_train "$@"
