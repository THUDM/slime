#!/usr/bin/env bash
# Scenario: linear (no sub-agents, no compact) - 2 rollout steps.
# Expected segment distribution: final 100% (no pre_wipe, no subagent).
# See SPEC §8.1 + §7.2 acceptance gates.

set -euo pipefail
source "$(dirname "$0")/run_qwen36_base.sh"

# Disable Agent / Task subagents + WebFetch / WebSearch.
export SWE_CLAUDE_EXTRA_ARGS="--disable-slash-commands --disallowedTools Agent Task WebFetch WebSearch"

# autoCompactWindow = 1_000_000 effectively disables autoCompact (no realistic
# claude-code session in this benchmark grows past 1M tokens).
cat > /tmp/cc_linear_settings.json <<'JSON'
{"autoCompactWindow": 1000000}
JSON
export SWE_CLAUDE_EXTRA_ARGS="${SWE_CLAUDE_EXTRA_ARGS} --settings /tmp/cc_linear_settings.json"

export NUM_STEPS_PER_ROLLOUT=2
exec_train "$@"
