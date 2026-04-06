from __future__ import annotations

from pathlib import Path


def test_multidomain_batch_submit_scripts_use_shared_submit_helper():
    examples_dir = Path(__file__).resolve().parents[1] / "examples" / "multidomain_v2"

    eval_submit = (examples_dir / "submit_eval_backfill.sh").read_text(encoding="utf-8")
    convert_submit = (examples_dir / "submit_convert_ckpt_to_hf.sh").read_text(encoding="utf-8")

    for script_text in (eval_submit, convert_submit):
        assert 'source "${SCRIPT_DIR}/../_shared/submit_inspire_utils.sh"' in script_text
        assert 'submit_inspire_command_job "${job_name}" 1 "${run_cmd}"' in script_text
        assert "job create" not in script_text
