"""Unit tests for MOPD (Multi-Teacher On-Policy Distillation).

Tests cover:
1. MOPD advantage computation (apply_mopd_to_advantages)
2. MOPD importance sampling weight computation and clipping
3. MOPD parameter validation in slime_validate_args
4. Sample.mopd_teacher_log_probs field
"""

import json
import os
import sys
import types
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helper to construct args for MOPD
# ---------------------------------------------------------------------------
def make_mopd_args(**overrides):
    """Create a Namespace with default MOPD arguments."""
    defaults = dict(
        use_mopd=True,
        mopd_teachers='[{"name": "math_teacher", "domain": "math"}]',
        mopd_teacher_loads=None,
        mopd_teacher_ckpt_steps=None,
        mopd_alpha=0.0,
        mopd_eps_low=0.2,
        mopd_eps_high=5.0,
        mopd_sampling_logprobs_key="rollout_log_probs",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# Tests for apply_mopd_to_advantages
# ---------------------------------------------------------------------------
class TestApplyMopdToAdvantages:
    """Test the apply_mopd_to_advantages function in loss.py."""

    @pytest.fixture(autouse=True)
    def _import_loss_module(self, monkeypatch):
        """Import loss.py with minimal megatron mocking."""
        # Mock megatron modules
        mpu_mod = types.ModuleType("megatron.core")
        mpu_sub = types.ModuleType("megatron.core.mpu")
        mpu_sub.is_pipeline_last_stage = lambda: True
        mpu_sub.get_context_parallel_rank = lambda: 0
        mpu_sub.get_context_parallel_world_size = lambda: 1
        mpu_sub.get_data_parallel_group = lambda: None
        mpu_sub.get_data_parallel_rank = lambda: 0
        mpu_sub.get_data_parallel_world_size = lambda: 1
        mpu_sub.get_tensor_model_parallel_rank = lambda: 0
        mpu_sub.get_tensor_model_parallel_world_size = lambda: 1

        monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
        monkeypatch.setitem(sys.modules, "megatron.core", mpu_mod)
        monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_sub)

    def _get_apply_mopd(self):
        """Dynamically import apply_mopd_to_advantages from loss.py."""
        from slime.backends.megatron_utils.loss import apply_mopd_to_advantages
        return apply_mopd_to_advantages

    def test_basic_mopd_advantage_computation(self):
        """Test that MOPD advantages are computed correctly with a single teacher."""
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args(mopd_alpha=0.0, mopd_eps_low=0.0, mopd_eps_high=1000.0)

        # Student log_probs: [0.1, -0.2, 0.3]
        # Teacher log_probs: [0.2, -0.1, 0.4]
        # reverse_kl = teacher - student = [0.1, 0.1, 0.1]
        student_log_probs = [torch.tensor([0.1, -0.2, 0.3])]
        teacher_log_probs = [torch.tensor([0.2, -0.1, 0.4])]

        # Sampling log_probs (μ_θ) = rollout_log_probs
        # IS weight = exp(student - sampling) = exp(student - rollout)
        # With rollout = student (same model), IS weight = 1.0 everywhere
        rollout_log_probs = [torch.tensor([0.1, -0.2, 0.3])]

        # Base advantages (ORM advantages)
        advantages = [torch.tensor([1.0, 2.0, 3.0])]

        rollout_data = {
            "mopd_teacher_log_probs": {"math": teacher_log_probs},
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd(args, rollout_data, advantages, student_log_probs)

        # With mopd_alpha=0: mopd_adv = reverse_kl = teacher - student = [0.1, 0.1, 0.1]
        # IS weight = exp(student - rollout) = exp(0) = 1.0, within bounds
        assert "mopd_advantages" in rollout_data
        assert "mopd_is_weights" in rollout_data
        assert "mopd_reverse_kl" in rollout_data

        # Check advantages (should not be modified in-place by MOPD, only stored)
        # Actually MOPD stores results in rollout_data, not modifying advantages
        mopd_adv = rollout_data["mopd_advantages"][0]
        is_weights = rollout_data["mopd_is_weights"][0]

        expected_reverse_kl = torch.tensor([0.1, 0.1, 0.1])
        assert torch.allclose(mopd_adv, expected_reverse_kl, atol=1e-6)
        assert torch.allclose(is_weights, torch.ones(3), atol=1e-6)

    # Check mopd_reverse_kl is pure reverse_kl (not including alpha * orm_advantage)
        reverse_kl_logged = rollout_data["mopd_reverse_kl"]["math"][0]
        expected_pure_reverse_kl = torch.tensor([0.1, 0.1, 0.1])
        assert torch.allclose(reverse_kl_logged, expected_pure_reverse_kl, atol=1e-6)

    def test_mopd_with_alpha(self):
        """Test MOPD with ORM advantage combination (alpha > 0)."""
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args(mopd_alpha=1.0, mopd_eps_low=0.0, mopd_eps_high=1000.0)

        student_log_probs = [torch.tensor([0.0, 0.0])]
        teacher_log_probs = [torch.tensor([1.0, 1.0])]
        rollout_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([2.0, 3.0])]

        rollout_data = {
            "mopd_teacher_log_probs": {"math": teacher_log_probs},
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd(args, rollout_data, advantages, student_log_probs)

        # reverse_kl = 1.0 - 0.0 = 1.0
        # mopd_adv = reverse_kl + alpha * ORM_adv = 1.0 + 1.0 * [2.0, 3.0] = [3.0, 4.0]
        mopd_adv = rollout_data["mopd_advantages"][0]
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(mopd_adv, expected, atol=1e-6)

        # mopd_reverse_kl should be pure reverse_kl, NOT containing alpha * orm_advantage
        reverse_kl_logged = rollout_data["mopd_reverse_kl"]["math"][0]
        expected_pure_reverse_kl = torch.tensor([1.0, 1.0])
        assert torch.allclose(reverse_kl_logged, expected_pure_reverse_kl, atol=1e-6)

    def test_is_weight_clipping_low(self):
        """Test that IS weights below eps_low are zeroed out."""
        apply_mopd = self._get_apply_mopd()
        # eps_low=0.5, so weights < 0.5 should be zeroed
        args = make_mopd_args(mopd_alpha=0.0, mopd_eps_low=0.5, mopd_eps_high=100.0)

        # student - rollout = very negative => IS weight = exp(very_negative) < 0.5
        student_log_probs = [torch.tensor([-5.0, 0.0])]
        teacher_log_probs = [torch.tensor([0.0, 0.0])]
        rollout_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([1.0, 1.0])]

        # IS weight for token 0: exp(-5.0 - 0.0) = exp(-5.0) ≈ 0.0067 < 0.5 -> zeroed
        # IS weight for token 1: exp(0.0 - 0.0) = 1.0 >= 0.5 -> kept
        rollout_data = {
            "mopd_teacher_log_probs": {"math": teacher_log_probs},
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd(args, rollout_data, advantages, student_log_probs)

        is_weights = rollout_data["mopd_is_weights"][0]
        assert is_weights[0].item() == 0.0, "IS weight below eps_low should be zeroed"
        assert is_weights[1].item() == 1.0, "IS weight within bounds should be kept"

    def test_is_weight_clipping_high(self):
        """Test that IS weights above eps_high are zeroed out."""
        apply_mopd = self._get_apply_mopd()
        # eps_high=2.0, so weights > 2.0 should be zeroed
        args = make_mopd_args(mopd_alpha=0.0, mopd_eps_low=0.0, mopd_eps_high=2.0)

        # student >> rollout => large IS weight
        student_log_probs = [torch.tensor([5.0, 0.0])]
        teacher_log_probs = [torch.tensor([0.0, 0.0])]
        rollout_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([1.0, 1.0])]

        # IS weight for token 0: exp(5.0 - 0.0) = exp(5.0) ≈ 148.4 > 2.0 -> zeroed
        # IS weight for token 1: exp(0.0 - 0.0) = 1.0 <= 2.0 -> kept
        rollout_data = {
            "mopd_teacher_log_probs": {"math": teacher_log_probs},
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd(args, rollout_data, advantages, student_log_probs)

        is_weights = rollout_data["mopd_is_weights"][0]
        assert is_weights[0].item() == 0.0, "IS weight above eps_high should be zeroed"
        assert is_weights[1].item() == 1.0, "IS weight within bounds should be kept"

    def test_multiple_teachers_averaged(self):
        """Test that MOPD advantages and IS weights are averaged across teachers."""
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args(mopd_alpha=0.0, mopd_eps_low=0.0, mopd_eps_high=1000.0)

        student_log_probs = [torch.tensor([0.0, 0.0])]
        # Two teachers with different log-probs
        teacher_math_log_probs = [torch.tensor([1.0, 1.0])]  # reverse_kl = [1.0, 1.0]
        teacher_code_log_probs = [torch.tensor([2.0, 2.0])]  # reverse_kl = [2.0, 2.0]
        rollout_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([0.0, 0.0])]

        rollout_data = {
            "mopd_teacher_log_probs": {
                "math": teacher_math_log_probs,
                "code": teacher_code_log_probs,
            },
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd(args, rollout_data, advantages, student_log_probs)

        # Averaged advantage = (1.0 + 2.0) / 2 = 1.5
        mopd_adv = rollout_data["mopd_advantages"][0]
        expected = torch.tensor([1.5, 1.5])
        assert torch.allclose(mopd_adv, expected, atol=1e-6)

        # IS weights should also be averaged (both are 1.0 here)
        is_weights = rollout_data["mopd_is_weights"][0]
        assert torch.allclose(is_weights, torch.ones(2), atol=1e-6)

    def test_per_sample_domain_routing(self):
        """Test per-sample domain routing with None entries in teacher_log_probs.

        When some samples don't have log-probs for a domain (None), that domain
        should be excluded from the average for those samples.
        """
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args(mopd_alpha=0.0, mopd_eps_low=0.0, mopd_eps_high=1000.0)

        # 2 samples, 2 teachers
        student_log_probs = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        # Sample 0: only math teacher (code is None)
        # Sample 1: both teachers
        teacher_math_log_probs = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
        teacher_code_log_probs = [None, torch.tensor([2.0, 2.0])]
        rollout_log_probs = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]

        rollout_data = {
            "mopd_teacher_log_probs": {
                "math": teacher_math_log_probs,
                "code": teacher_code_log_probs,
            },
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd(args, rollout_data, advantages, student_log_probs)

        # Sample 0: only math teacher (reverse_kl = 1.0), so mopd_adv = 1.0
        mopd_adv_s0 = rollout_data["mopd_advantages"][0]
        assert torch.allclose(mopd_adv_s0, torch.tensor([1.0, 1.0]), atol=1e-6)

        # Sample 1: both teachers, averaged = (1.0 + 2.0) / 2 = 1.5
        mopd_adv_s1 = rollout_data["mopd_advantages"][1]
        assert torch.allclose(mopd_adv_s1, torch.tensor([1.5, 1.5]), atol=1e-6)

    def test_per_sample_all_domains_none(self):
        """Test that a sample with no valid domains gets zero advantages and IS weights."""
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args(mopd_alpha=0.0, mopd_eps_low=0.0, mopd_eps_high=1000.0)

        student_log_probs = [torch.tensor([0.0, 0.0])]
        # All domains are None for this sample
        teacher_math_log_probs = [None]
        teacher_code_log_probs = [None]
        rollout_log_probs = [torch.tensor([0.0, 0.0])]
        advantages = [torch.tensor([1.0, 1.0])]

        rollout_data = {
            "mopd_teacher_log_probs": {
                "math": teacher_math_log_probs,
                "code": teacher_code_log_probs,
            },
            "rollout_log_probs": rollout_log_probs,
        }

        apply_mopd(args, rollout_data, advantages, student_log_probs)

        # Should get zeros since no valid teachers
        mopd_adv = rollout_data["mopd_advantages"][0]
        assert torch.allclose(mopd_adv, torch.zeros(2), atol=1e-6)
        is_weights = rollout_data["mopd_is_weights"][0]
        assert torch.allclose(is_weights, torch.zeros(2), atol=1e-6)

    def test_student_log_probs_none_returns_early(self):
        """Test that apply_mopd returns early when student_log_probs is None."""
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args()
        advantages = [torch.tensor([1.0])]

        rollout_data = {
            "mopd_teacher_log_probs": {"math": [torch.tensor([1.0])]},
        }

        # Should not raise, just return early
        apply_mopd(args, rollout_data, advantages, None)
        # No MOPD keys should be added
        assert "mopd_advantages" not in rollout_data

    def test_missing_teacher_log_probs_raises(self):
        """Test that missing mopd_teacher_log_probs raises ValueError."""
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args()
        student_log_probs = [torch.tensor([0.0])]
        advantages = [torch.tensor([1.0])]
        rollout_data = {}

        with pytest.raises(ValueError, match="mopd_teacher_log_probs"):
            apply_mopd(args, rollout_data, advantages, student_log_probs)

    def test_empty_teacher_log_probs_raises(self):
        """Test that empty mopd_teacher_log_probs dict raises ValueError."""
        apply_mopd = self._get_apply_mopd()
        args = make_mopd_args()
        student_log_probs = [torch.tensor([0.0])]
        advantages = [torch.tensor([1.0])]
        rollout_data = {"mopd_teacher_log_probs": {}}

        with pytest.raises(ValueError, match="mopd_teacher_log_probs"):
            apply_mopd(args, rollout_data, advantages, student_log_probs)


# ---------------------------------------------------------------------------
# Tests for MOPD parameter validation
# ---------------------------------------------------------------------------
class TestMopdArgValidation:
    """Test MOPD argument validation in slime_validate_args."""

    @pytest.fixture(autouse=True)
    def _mock_deps(self, monkeypatch):
        """Mock megatron and other dependencies for arguments module."""
        megatron_mod = types.ModuleType("megatron")
        training_mod = types.ModuleType("megatron.training")
        arguments_mod = types.ModuleType("megatron.training.arguments")
        arguments_mod.parse_args = lambda *a, **kw: None
        arguments_mod.validate_args = lambda a: a
        tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
        tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
        tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *a, **kw: None)

        monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
        monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
        monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
        monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
        monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    def _make_base_args(self, **overrides):
        """Create a minimal valid args Namespace for validation tests."""
        defaults = dict(
            use_opd=False,
            opd_type=None,
            opd_kl_coef=1.0,
            opd_teacher_load=None,
            opd_teacher_ckpt_step=None,
            use_mopd=False,
            mopd_teachers=None,
            mopd_teacher_loads=None,
            mopd_teacher_ckpt_steps=None,
            mopd_alpha=0.0,
            mopd_eps_low=0.2,
            mopd_eps_high=5.0,
            mopd_sampling_logprobs_key="rollout_log_probs",
            enable_weights_backuper=True,
            eval_datasets=[],
            eval_prompt_data=None,
            kl_coef=0,
            ref_load="/tmp/fake_ref",
            use_kl_loss=False,
            use_critic=False,
            rm_type=None,
            custom_rm_path=None,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_mopd_and_opd_mutually_exclusive(self):
        """Test that --use-mopd and --use-opd cannot be used together."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(use_mopd=True, use_opd=True)
        with pytest.raises(ValueError, match="mutually exclusive"):
            slime_validate_args(args)

    def test_mopd_requires_teachers(self):
        """Test that --use-mopd requires --mopd-teachers."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(use_mopd=True, mopd_teachers=None)
        with pytest.raises(ValueError, match="mopd-teachers"):
            slime_validate_args(args)

    def test_mopd_duplicate_domain_raises(self):
        """Test that duplicate domains in --mopd-teachers raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}, {"name": "t2", "domain": "math"}]',
        )
        with pytest.raises(ValueError, match="duplicate domain"):
            slime_validate_args(args)

    def test_mopd_missing_domain_raises(self):
        """Test that teacher config without 'domain' key raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1"}]',
        )
        with pytest.raises(ValueError, match="domain"):
            slime_validate_args(args)

    def test_mopd_eps_low_negative_raises(self):
        """Test that negative eps_low raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_eps_low=-0.1,
        )
        with pytest.raises(ValueError, match="mopd-eps-low"):
            slime_validate_args(args)

    def test_mopd_eps_high_leq_eps_low_raises(self):
        """Test that eps_high <= eps_low raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_eps_low=5.0,
            mopd_eps_high=5.0,
        )
        with pytest.raises(ValueError, match="mopd-eps-high"):
            slime_validate_args(args)

    def test_mopd_teacher_loads_count_mismatch(self):
        """Test that mismatched mopd_teacher_loads count raises ValueError."""
        from slime.utils.arguments import slime_validate_args, tmp_path

        # Create fake checkpoint dirs
        ckpt_dir = tmp_path / "teacher1"
        ckpt_dir.mkdir()
        (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("1")

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}, {"name": "t2", "domain": "code"}]',
            mopd_teacher_loads=[str(ckpt_dir)],  # 1 path but 2 teachers
        )
        with pytest.raises(ValueError, match="mopd-teacher-loads"):
            slime_validate_args(args)

    def test_mopd_not_enabled_loads_set_raises(self):
        """Test that mopd_teacher_loads without --use-mopd raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(use_mopd=False, mopd_teacher_loads=["/tmp/fake"])
        with pytest.raises(ValueError, match="use-mopd is not enabled"):
            slime_validate_args(args)

    def test_mopd_alpha_gt0_without_rm_type_raises(self):
        """Test that --mopd-alpha > 0 without --rm-type or --custom-rm-path raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_alpha=1.0,
            rm_type=None,
            custom_rm_path=None,
        )
        with pytest.raises(ValueError, match="mopd-alpha > 0 requires a reward model"):
            slime_validate_args(args)

    def test_mopd_alpha_zero_without_rm_type_defaults_to_zero(self):
        """Test that --mopd-alpha 0 without --rm-type defaults rm_type to 'zero'."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_alpha=0.0,
            rm_type=None,
            custom_rm_path=None,
        )
        slime_validate_args(args)
        assert args.rm_type == "zero"

    def test_mopd_alpha_gt0_with_rm_type_ok(self):
        """Test that --mopd-alpha > 0 with --rm-type does not raise."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_alpha=1.0,
            rm_type="math",
            custom_rm_path=None,
        )
        slime_validate_args(args)  # Should not raise

    def test_mopd_alpha_gt0_with_custom_rm_ok(self):
        """Test that --mopd-alpha > 0 with --custom-rm-path does not raise."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_alpha=1.0,
            rm_type=None,
            custom_rm_path="some.module.func",
        )
        slime_validate_args(args)  # Should not raise


# ---------------------------------------------------------------------------
# Tests for Sample.mopd_teacher_log_probs field
# ---------------------------------------------------------------------------
class TestSampleMopdField:
    """Test that Sample supports mopd_teacher_log_probs field."""

    def test_default_none(self):
        from slime.utils.types import Sample
        s = Sample()
        assert s.mopd_teacher_log_probs is None

    def test_set_mopd_teacher_log_probs(self):
        from slime.utils.types import Sample
        import torch
        s = Sample()
        s.mopd_teacher_log_probs = {
            "math": torch.tensor([0.1, 0.2, 0.3]),
            "code": torch.tensor([0.4, 0.5, 0.6]),
        }
        assert "math" in s.mopd_teacher_log_probs
        assert "code" in s.mopd_teacher_log_probs
        assert len(s.mopd_teacher_log_probs["math"]) == 3

    def test_to_dict_roundtrip(self):
        from slime.utils.types import Sample
        s = Sample(response="hello", response_length=1)
        s.mopd_teacher_log_probs = {"math": [0.1, 0.2, 0.3]}
        d = s.to_dict()
        assert "mopd_teacher_log_probs" in d
        assert d["mopd_teacher_log_probs"]["math"] == [0.1, 0.2, 0.3]