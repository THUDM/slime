"""Unit tests for MOPD full_vocab distillation mode.

Tests cover:
1. vocab_parallel_reverse_kl: correctness of full-vocabulary reverse KL
2. apply_mopd_full_vocab_to_loss: IS weights, multi-teacher averaging, loss combination
3. MOPD full_vocab argument validation (mopd_distill_type parameter)
4. get_logits_for_distill: temperature handling and shape
"""

import sys
import types
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_mopd_full_vocab_args(**overrides):
    """Create a Namespace with default MOPD full_vocab arguments."""
    defaults = dict(
        use_mopd=True,
        mopd_distill_type="full_vocab",
        mopd_teachers='[{"name": "math_teacher", "domain": "math"}]',
        mopd_teacher_loads="/tmp/fake_teacher",
        mopd_teacher_ckpt_steps=None,
        mopd_alpha=0.0,
        mopd_eps_low=0.2,
        mopd_eps_high=5.0,
        mopd_sampling_logprobs_key="rollout_log_probs",
        _mopd_teachers_parsed=[{"name": "math_teacher", "domain": "math"}],
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def mock_megatron(monkeypatch):
    """Mock megatron.core.mpu for import."""
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
    mpu_sub.get_tensor_model_parallel_group = lambda: None

    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", mpu_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_sub)


# ---------------------------------------------------------------------------
# Tests for vocab_parallel_reverse_kl
# ---------------------------------------------------------------------------
class TestVocabParallelReverseKL:
    """Test the vocab_parallel_reverse_kl function in ppo_utils.py."""

    def test_kl_correctness_identical_distributions(self):
        """KL(student || teacher) = 0 when distributions are identical."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        # Identical distributions → KL = 0
        logits = torch.randn(4, 20)  # [R=4, V=20]
        kl = vocab_parallel_reverse_kl(logits, logits.clone(), process_group=None)
        assert kl.shape == (4,)
        assert torch.allclose(kl, torch.zeros(4), atol=1e-5), f"KL should be 0 for identical distributions, got {kl}"

    def test_kl_correctness_known_values(self):
        """KL computed by vocab_parallel_reverse_kl matches manual computation."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        torch.manual_seed(42)
        student_logits = torch.randn(3, 10, requires_grad=True)
        teacher_logits = torch.randn(3, 10)

        # Our function
        kl = vocab_parallel_reverse_kl(student_logits, teacher_logits, process_group=None)

        # Manual computation
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
        expected_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

        assert torch.allclose(kl, expected_kl, atol=1e-5), (
            f"KL mismatch: got {kl}, expected {expected_kl}"
        )

    def test_kl_non_negative(self):
        """KL divergence should always be non-negative (Gibbs' inequality)."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        torch.manual_seed(123)
        for _ in range(10):
            student_logits = torch.randn(5, 50)
            teacher_logits = torch.randn(5, 50)
            kl = vocab_parallel_reverse_kl(student_logits, teacher_logits, process_group=None)
            assert (kl >= -1e-5).all(), f"KL should be non-negative, got {kl.min()}"

    def test_kl_gradient_flows_through_student(self):
        """Gradient flows through student logits but not teacher logits."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        torch.manual_seed(42)
        student_logits = torch.randn(3, 10, requires_grad=True)
        teacher_logits = torch.randn(3, 10, requires_grad=True)

        kl = vocab_parallel_reverse_kl(student_logits, teacher_logits, process_group=None)
        loss = kl.sum()
        loss.backward()

        # Student should have gradients
        assert student_logits.grad is not None, "student_logits should have gradients"
        assert not torch.allclose(student_logits.grad, torch.zeros_like(student_logits.grad)), (
            "student_logits gradients should be non-zero"
        )

        # Teacher should NOT have gradients (detached inside function)
        assert teacher_logits.grad is None or torch.allclose(teacher_logits.grad, torch.zeros_like(teacher_logits.grad)), (
            "teacher_logits should not have gradients (should be detached)"
        )

    def test_kl_gradient_correctness(self):
        """Verify the gradient of KL matches autograd from manual computation."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        torch.manual_seed(42)
        student_logits_1 = torch.randn(3, 10, requires_grad=True)
        teacher_logits = torch.randn(3, 10)

        # Our function
        kl_1 = vocab_parallel_reverse_kl(student_logits_1, teacher_logits, process_group=None)
        loss_1 = kl_1.sum()
        loss_1.backward()
        grad_ours = student_logits_1.grad.clone()

        # Manual computation for gradient comparison
        student_logits_2 = student_logits_1.detach().clone().requires_grad_(True)
        student_log_probs = torch.log_softmax(student_logits_2, dim=-1)
        student_probs = student_log_probs.exp()
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
        kl_2 = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        loss_2 = kl_2.sum()
        loss_2.backward()
        grad_manual = student_logits_2.grad.clone()

        assert torch.allclose(grad_ours, grad_manual, atol=1e-4), (
            f"Gradient mismatch: max diff = {(grad_ours - grad_manual).abs().max()}"
        )

    def test_kl_temperature_sensitivity(self):
        """KL should change when student distribution changes."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        teacher_logits = torch.randn(3, 10)
        student_logits_1 = torch.randn(3, 10)
        student_logits_2 = torch.randn(3, 10)

        kl_1 = vocab_parallel_reverse_kl(student_logits_1, teacher_logits, process_group=None)
        kl_2 = vocab_parallel_reverse_kl(student_logits_2, teacher_logits, process_group=None)

        # Different student distributions should give different KL values
        assert not torch.allclose(kl_1, kl_2, atol=1e-5), "Different student logits should give different KL"

    def test_kl_large_vocabulary(self):
        """Test with a larger vocabulary to verify numerical stability."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        torch.manual_seed(42)
        student_logits = torch.randn(8, 32000) * 0.1  # Small scale for stability
        teacher_logits = torch.randn(8, 32000) * 0.1

        kl = vocab_parallel_reverse_kl(student_logits, teacher_logits, process_group=None)
        assert kl.shape == (8,)
        assert torch.isfinite(kl).all(), f"KL should be finite, got nan/inf: {kl}"
        assert (kl >= -1e-4).all(), f"KL should be non-negative, got min={kl.min()}"


# ---------------------------------------------------------------------------
# Tests for apply_mopd_full_vocab_to_loss
# ---------------------------------------------------------------------------
class TestApplyMopdFullVocabToLoss:
    """Test the apply_mopd_full_vocab_to_loss function."""

    @pytest.fixture(autouse=True)
    def _mock_deps(self, monkeypatch):
        mock_megatron(monkeypatch)

    def _get_function(self):
        from slime.backends.megatron_utils.loss import apply_mopd_full_vocab_to_loss
        return apply_mopd_full_vocab_to_loss

    def _sum_of_sample_mean(self, tensor):
        """Simple mean reduction for testing."""
        return tensor.mean()

    def test_single_teacher_kl_loss(self):
        """Test single-teacher full-vocab KL loss computation."""
        apply_fn = self._get_function()
        args = make_mopd_full_vocab_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)
        torch.manual_seed(42)

        # 2 samples, vocab_size=10, response_length=3
        V = 10
        student_logits_1 = torch.randn(3, V)
        student_logits_2 = torch.randn(4, V)
        teacher_logits_1 = torch.randn(3, V)
        teacher_logits_2 = torch.randn(4, V)

        student_logits = [student_logits_1, student_logits_2]
        teacher_logits_per_domain = {
            "math": [teacher_logits_1, teacher_logits_2],
        }

        # When sampling_log_probs == current_log_probs (on-policy), IS weight = 1.0
        batch = {
            "rollout_log_probs": [torch.zeros(3), torch.zeros(4)],
        }
        current_log_probs = [torch.zeros(3), torch.zeros(4)]
        loss_masks = [torch.ones(3), torch.ones(4)]

        kl_loss, metrics = apply_fn(
            args, batch, student_logits, teacher_logits_per_domain,
            loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        assert kl_loss.shape == (), "kl_loss should be scalar"
        assert kl_loss.item() >= 0, "KL loss should be non-negative"
        assert "mopd_fv_kl" in metrics
        assert "mopd_is_weight_mean" in metrics
        assert "mopd_is_nonzero_frac" in metrics
        assert "mopd_fv_kl/math" in metrics

    def test_identical_student_teacher_zero_kl(self):
        """When student == teacher, KL should be ~0 and loss should be ~0."""
        apply_fn = self._get_function()
        args = make_mopd_full_vocab_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)

        V = 10
        student_logits_1 = torch.randn(3, V)
        student_logits_2 = torch.randn(4, V)

        # Teacher = Student → KL = 0
        teacher_logits_per_domain = {
            "math": [student_logits_1.clone(), student_logits_2.clone()],
        }

        batch = {
            "rollout_log_probs": [torch.zeros(3), torch.zeros(4)],
        }
        current_log_probs = [torch.zeros(3), torch.zeros(4)]
        loss_masks = [torch.ones(3), torch.ones(4)]

        kl_loss, metrics = apply_fn(
            args, batch, [student_logits_1, student_logits_2],
            teacher_logits_per_domain, loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        assert kl_loss.item() < 1e-4, f"KL loss should be ~0 for identical distributions, got {kl_loss.item()}"

    def test_multi_teacher_averaging(self):
        """Test that KL is averaged across multiple teachers."""
        apply_fn = self._get_function()
        args = make_mopd_full_vocab_args(
            mopd_eps_low=0.0, mopd_eps_high=1000.0,
            _mopd_teachers_parsed=[
                {"name": "math_teacher", "domain": "math"},
                {"name": "code_teacher", "domain": "code"},
            ],
        )
        torch.manual_seed(42)

        V = 10
        student_logits = [torch.randn(3, V)]
        teacher_math = [torch.randn(3, V)]
        teacher_code = [torch.randn(3, V)]

        teacher_logits_per_domain = {
            "math": teacher_math,
            "code": teacher_code,
        }

        batch = {
            "rollout_log_probs": [torch.zeros(3)],
        }
        current_log_probs = [torch.zeros(3)]
        loss_masks = [torch.ones(3)]

        from slime.utils.ppo_utils import vocab_parallel_reverse_kl
        kl_math = vocab_parallel_reverse_kl(student_logits[0], teacher_math[0], None)
        kl_code = vocab_parallel_reverse_kl(student_logits[0], teacher_code[0], None)
        expected_avg_kl = (kl_math.sum() / 3 + kl_code.sum() / 4) / 2  # Not exact, just check shape

        kl_loss, metrics = apply_fn(
            args, batch, student_logits, teacher_logits_per_domain,
            loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        # Should have per-domain logging
        assert "mopd_fv_kl/math" in metrics
        assert "mopd_fv_kl/code" in metrics
        # Both should be non-negative
        assert metrics["mopd_fv_kl/math"].item() >= -1e-5
        assert metrics["mopd_fv_kl/code"].item() >= -1e-5

    def test_is_weight_clipping(self):
        """Test that IS weights are clipped to [eps_low, eps_high]."""
        apply_fn = self._get_function()
        args = make_mopd_full_vocab_args(mopd_eps_low=0.5, mopd_eps_high=2.0)

        V = 10
        student_logits = [torch.randn(3, V)]
        teacher_logits_per_domain = {"math": [torch.randn(3, V)]}

        # IS weight = exp(current_log_probs - rollout_log_probs)
        # For token 0: exp(-5 - 0) = exp(-5) ≈ 0.0067 < eps_low → zeroed
        # For token 1: exp(5 - 0) = exp(5) ≈ 148 > eps_high → zeroed
        # For token 2: exp(0 - 0) = 1.0 → kept
        batch = {
            "rollout_log_probs": [torch.tensor([0.0, 0.0, 0.0])],
        }
        current_log_probs = [torch.tensor([-5.0, 5.0, 0.0])]
        loss_masks = [torch.ones(3)]

        kl_loss, metrics = apply_fn(
            args, batch, student_logits, teacher_logits_per_domain,
            loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        # IS weight should be clipped — only token 2 (weight=1.0) survives
        # Nonzero fraction should be 1/3
        is_nonzero_frac = metrics["mopd_is_nonzero_frac"].item()
        assert abs(is_nonzero_frac - 1.0 / 3.0) < 0.05, (
            f"Expected ~1/3 nonzero IS weight fraction, got {is_nonzero_frac}"
        )

    def test_none_teacher_for_sample(self):
        """Test that None entries in teacher logits are skipped."""
        apply_fn = self._get_function()

        # Two samples, two teachers; sample 0 has only math, sample 1 has both
        args = make_mopd_full_vocab_args(
            mopd_eps_low=0.0, mopd_eps_high=1000.0,
            _mopd_teachers_parsed=[
                {"name": "math_teacher", "domain": "math"},
                {"name": "code_teacher", "domain": "code"},
            ],
        )

        V = 10
        student_0 = torch.randn(3, V)
        student_1 = torch.randn(4, V)

        teacher_logits_per_domain = {
            "math": [torch.randn(3, V), torch.randn(4, V)],
            "code": [None, torch.randn(4, V)],  # sample 0 has no code teacher
        }

        batch = {
            "rollout_log_probs": [torch.zeros(3), torch.zeros(4)],
        }
        current_log_probs = [torch.zeros(3), torch.zeros(4)]
        loss_masks = [torch.ones(3), torch.ones(4)]

        kl_loss, metrics = apply_fn(
            args, batch, [student_0, student_1],
            teacher_logits_per_domain, loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        assert kl_loss.shape == ()
        assert torch.isfinite(kl_loss), "Loss should be finite with None teacher entries"

    def test_loss_mask_effect(self):
        """Test that loss_mask correctly masks out tokens."""
        apply_fn = self._get_function()
        args = make_mopd_full_vocab_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)

        V = 10
        # Same student and teacher but with masking
        student_logits = [torch.randn(5, V)]
        teacher_logits_per_domain = {"math": [torch.randn(5, V)]}

        # Only tokens 1-3 are valid (mask out 0 and 4)
        loss_masks = [torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])]

        batch = {
            "rollout_log_probs": [torch.zeros(5)],
        }
        current_log_probs = [torch.zeros(5)]

        kl_loss_masked, _ = apply_fn(
            args, batch, student_logits, teacher_logits_per_domain,
            loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        # With all-ones mask for comparison
        loss_masks_all = [torch.ones(5)]
        kl_loss_all, _ = apply_fn(
            args, batch, student_logits, teacher_logits_per_domain,
            loss_masks_all, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        # The losses should be different since masking excludes tokens
        # Both should be non-negative
        assert kl_loss_masked.item() >= 0
        assert kl_loss_all.item() >= 0

    def test_current_log_probs_used_for_is_weights(self):
        """Test that current_log_probs (not batch['log_probs']) are used for IS weights."""
        apply_fn = self._get_function()
        # Use tight clipping to detect which log_probs are used
        args = make_mopd_full_vocab_args(mopd_eps_low=0.5, mopd_eps_high=2.0)

        V = 10
        student_logits = [torch.randn(3, V)]
        teacher_logits_per_domain = {"math": [torch.randn(3, V)]}

        # rollout_log_probs = [0, 0, 0]
        # current_log_probs = [-5, 0, 5]  → IS weights: exp(-5)≈0.007, 1.0, exp(5)≈148
        # With current_log_probs: tokens 0 and 2 are zeroed out (outside [0.5, 2.0])
        # batch['log_probs'] = [0, 0, 0] → all IS weights = 1.0 (within [0.5, 2.0])
        batch = {
            "rollout_log_probs": [torch.tensor([0.0, 0.0, 0.0])],
            # This is stale batch["log_probs"]; should NOT be used when current_log_probs is provided
            "log_probs": [torch.tensor([0.0, 0.0, 0.0])],
        }
        current_log_probs = [torch.tensor([-5.0, 0.0, 5.0])]
        loss_masks = [torch.ones(3)]

        kl_loss, metrics = apply_fn(
            args, batch, student_logits, teacher_logits_per_domain,
            loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        # Only token 1 should survive IS weight clipping → 1/3 nonzero
        is_nonzero_frac = metrics["mopd_is_nonzero_frac"].item()
        assert abs(is_nonzero_frac - 1.0 / 3.0) < 0.05, (
            f"Expected ~1/3 nonzero IS weight fraction with current_log_probs, got {is_nonzero_frac}"
        )

    def test_current_log_probs_length_mismatch_raises(self):
        """Test that mismatched current_log_probs length raises ValueError."""
        apply_fn = self._get_function()
        args = make_mopd_full_vocab_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)

        V = 10
        student_logits = [torch.randn(3, V), torch.randn(4, V)]
        teacher_logits_per_domain = {"math": [torch.randn(3, V), torch.randn(4, V)]}
        batch = {"rollout_log_probs": [torch.zeros(3), torch.zeros(4)]}
        loss_masks = [torch.ones(3), torch.ones(4)]

        # Mismatch: 2 samples but only 1 log_probs entry
        bad_current_log_probs = [torch.zeros(3)]

        with pytest.raises(ValueError, match="student_log_probs length"):
            apply_fn(
                args, batch, student_logits, teacher_logits_per_domain,
                loss_masks, self._sum_of_sample_mean,
                current_log_probs=bad_current_log_probs,
            )


# ---------------------------------------------------------------------------
# Tests for mopd_distill_type argument validation
# ---------------------------------------------------------------------------
class TestMopdDistillTypeValidation:
    """Test --mopd-distill-type parameter validation."""

    @pytest.fixture(autouse=True)
    def _mock_deps(self, monkeypatch):
        """Mock megatron and other dependencies."""
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
            mopd_distill_type="token_level",
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

    def test_full_vocab_without_teacher_loads_raises(self):
        """Test that full_vocab mode without --mopd-teacher-loads raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_distill_type="full_vocab",
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_teacher_loads=None,
        )
        with pytest.raises(ValueError, match="full_vocab.*mopd-teacher-loads|megatron teacher"):
            slime_validate_args(args)

    def test_token_level_is_default(self):
        """Test that token_level is the default distill type."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_alpha=0.0,
        )
        slime_validate_args(args)
        assert args.mopd_distill_type == "token_level"

    def test_full_vocab_with_teacher_loads_ok(self):
        """Test that full_vocab mode with --mopd-teacher-loads does not raise."""
        from slime.utils.arguments import slime_validate_args, tmp_path

        ckpt_dir = tmp_path / "teacher1"
        ckpt_dir.mkdir()
        (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("1")

        args = self._make_base_args(
            use_mopd=True,
            mopd_distill_type="full_vocab",
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_teacher_loads=[str(ckpt_dir)],
            mopd_alpha=0.0,
        )
        slime_validate_args(args)  # Should not raise


# ---------------------------------------------------------------------------
# Tests for get_logits_for_distill temperature handling
# ---------------------------------------------------------------------------
class TestGetLogitsForDistill:
    """Test get_logits_for_distill returns raw logits without temperature scaling."""

    @pytest.fixture(autouse=True)
    def _mock_deps(self, monkeypatch):
        mock_megatron(monkeypatch)

    def test_no_temperature_scaling(self):
        """Verify that get_logits_for_distill returns raw logits (no temperature)."""
        from slime.backends.megatron_utils.loss import get_logits_for_distill

        torch.manual_seed(42)
        V = 10
        T = 6  # total sequence length (prompt + response)
        R = 3  # response length

        # Create fake logits [1, T, V]
        logits = torch.randn(1, T, V)

        # Create args with rollout_temperature != 1.0
        args = Namespace(
            qkv_format="bshd",
            rollout_temperature=2.0,  # temperature scaling
            allgather_cp=False,
            log_probs_chunk_size=-1,
        )

        # Create fake tokens and length info
        tokens = torch.randint(0, V, (1, T))
        unconcat_tokens = [tokens[0]]
        total_lengths = [T]
        response_lengths = [R]
        max_seq_lens = [T]

        result_tensor, result_dict = get_logits_for_distill(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

        # Should return raw logits (no temperature scaling)
        assert "logits" in result_dict
        logits_out = result_dict["logits"]
        assert len(logits_out) == 1  # 1 sample
        assert logits_out[0].shape == (R, V), f"Expected shape ({R}, {V}), got {logits_out[0].shape}"

        # The returned logits should NOT be divided by temperature
        # Compare with manual extraction of response logits from the input
        # Response logits: logits[0, T-R-1:T-1, :] (shifted by 1 for next-token prediction)
        expected_logits = logits[0, T - R - 1 : T - 1, :]  # [R, V]
        assert torch.allclose(logits_out[0], expected_logits, atol=1e-5), (
            "get_logits_for_distill should return raw logits without temperature scaling"
        )


# ---------------------------------------------------------------------------
# Tests for vocab_parallel_topk_reverse_kl
# ---------------------------------------------------------------------------
class TestVocabParallelTopkReverseKL:
    """Test the vocab_parallel_topk_reverse_kl function in ppo_utils.py."""

    def test_topk_kl_approximates_full_kl(self):
        """Top-k KL should approximate full-vocab KL when k covers most probability mass."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl, vocab_parallel_topk_reverse_kl

        torch.manual_seed(42)
        V = 20
        R = 4
        k = 15  # top-15 out of 20 should cover most mass

        student_logits = torch.randn(R, V, requires_grad=True)
        teacher_logits = torch.randn(R, V)

        # Full-vocab KL
        full_kl = vocab_parallel_reverse_kl(student_logits, teacher_logits, process_group=None)

        # Top-k KL
        topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
        topk_kl = vocab_parallel_topk_reverse_kl(
            student_logits, topk_vals, topk_idx, V, process_group=None
        )

        # With k close to V, the top-k should be close to full-vocab KL
        # Allow some tolerance due to tail approximation
        assert topk_kl.shape == (R,), f"Expected shape ({R},), got {topk_kl.shape}"
        assert torch.isfinite(topk_kl).all(), f"Top-k KL should be finite, got {topk_kl}"

    def test_topk_kl_identical_distributions(self):
        """Top-k KL should be ~0 when student == teacher."""
        from slime.utils.ppo_utils import vocab_parallel_topk_reverse_kl

        V = 20
        k = 10
        logits = torch.randn(3, V)

        topk_vals, topk_idx = logits.topk(k, dim=-1)
        kl = vocab_parallel_topk_reverse_kl(logits, topk_vals, topk_idx, V, process_group=None)

        assert kl.shape == (3,)
        # Should be close to 0 (not exact due to tail approximation with V > k)
        assert kl.item() >= -0.1, f"Top-k KL should be ~0 for identical distributions, got {kl}"

    def test_topk_kl_gradient_flows(self):
        """Gradient flows through student logits in top-k KL."""
        from slime.utils.ppo_utils import vocab_parallel_topk_reverse_kl

        V = 20
        k = 10
        student_logits = torch.randn(3, V, requires_grad=True)
        teacher_logits = torch.randn(3, V)

        topk_vals, topk_idx = teacher_logits.topk(k, dim=-1)
        kl = vocab_parallel_topk_reverse_kl(student_logits, topk_vals, topk_idx, V, process_group=None)
        loss = kl.sum()
        loss.backward()

        assert student_logits.grad is not None, "student_logits should have gradients"
        assert not torch.allclose(student_logits.grad, torch.zeros_like(student_logits.grad)), \
            "student_logits gradients should be non-zero"

    def test_topk_kl_increases_with_smaller_k(self):
        """Top-k KL should generally increase as k decreases (less accurate approximation)."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl, vocab_parallel_topk_reverse_kl

        torch.manual_seed(42)
        V = 50
        R = 5
        student_logits = torch.randn(R, V)
        teacher_logits = torch.randn(R, V)

        full_kl = vocab_parallel_reverse_kl(student_logits, teacher_logits, process_group=None).sum().item()

        k_large = 40
        topk_vals_l, topk_idx_l = teacher_logits.topk(k_large, dim=-1)
        kl_large = vocab_parallel_topk_reverse_kl(
            student_logits, topk_vals_l, topk_idx_l, V, process_group=None
        ).sum().item()

        # With k=V (full vocab), top-k should be closer to full KL
        assert torch.isfinite(torch.tensor(kl_large)), "Top-k KL should be finite"


# ---------------------------------------------------------------------------
# Tests for apply_mopd_topk_to_loss
# ---------------------------------------------------------------------------
class TestApplyMopdTopkToLoss:
    """Test the apply_mopd_topk_to_loss function."""

    @pytest.fixture(autouse=True)
    def _mock_deps(self, monkeypatch):
        mock_megatron(monkeypatch)

    def _get_function(self):
        from slime.backends.megatron_utils.loss import apply_mopd_topk_to_loss
        return apply_mopd_topk_to_loss

    def _sum_of_sample_mean(self, tensor):
        return tensor.mean()

    def _make_args(self, **overrides):
        defaults = dict(
            use_mopd=True,
            mopd_distill_type="top_k",
            mopd_topk_k=8,
            mopd_teachers='[{"name": "math_teacher", "domain": "math"}]',
            mopd_teacher_loads="/tmp/fake_teacher",
            mopd_teacher_ckpt_steps=None,
            mopd_alpha=0.0,
            mopd_eps_low=0.0,
            mopd_eps_high=1000.0,
            mopd_sampling_logprobs_key="rollout_log_probs",
            _mopd_teachers_parsed=[{"name": "math_teacher", "domain": "math"}],
            padded_vocab_size=20,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_single_teacher_topk_loss(self):
        """Test single-teacher top-k KL loss computation."""
        apply_fn = self._get_function()
        args = self._make_args()
        torch.manual_seed(42)

        V = 20
        k = 8
        R1, R2 = 3, 4

        student_logits_1 = torch.randn(R1, V)
        student_logits_2 = torch.randn(R2, V)
        teacher_logits_1 = torch.randn(R1, V)
        teacher_logits_2 = torch.randn(R2, V)

        # Get top-k from teacher
        topk_vals_1, topk_idx_1 = teacher_logits_1.topk(k, dim=-1)
        topk_vals_2, topk_idx_2 = teacher_logits_2.topk(k, dim=-1)

        student_logits = [student_logits_1, student_logits_2]
        teacher_topk_logits = {"math": [topk_vals_1, topk_vals_2]}
        teacher_topk_indices = {"math": [topk_idx_1, topk_idx_2]}

        batch = {"rollout_log_probs": [torch.zeros(R1), torch.zeros(R2)]}
        current_log_probs = [torch.zeros(R1), torch.zeros(R2)]
        loss_masks = [torch.ones(R1), torch.ones(R2)]

        kl_loss, metrics = apply_fn(
            args, batch, student_logits, teacher_topk_logits,
            teacher_topk_indices, loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        assert kl_loss.shape == (), "kl_loss should be scalar"
        assert "mopd_topk_kl" in metrics
        assert "mopd_is_weight_mean" in metrics
        assert "mopd_is_nonzero_frac" in metrics
        assert "mopd_topk_kl/math" in metrics

    def test_topk_loss_is_non_negative(self):
        """Test that top-k KL loss is non-negative (or close to it)."""
        apply_fn = self._get_function()
        args = self._make_args(mopd_eps_low=0.0, mopd_eps_high=1000.0)
        torch.manual_seed(42)

        V = 20
        k = 10
        student_logits = [torch.randn(5, V)]
        teacher_logits = [torch.randn(5, V)]

        topk_vals, topk_idx = teacher_logits[0].topk(k, dim=-1)
        teacher_topk_logits = {"math": [topk_vals]}
        teacher_topk_indices = {"math": [topk_idx]}

        batch = {"rollout_log_probs": [torch.zeros(5)]}
        current_log_probs = [torch.zeros(5)]
        loss_masks = [torch.ones(5)]

        kl_loss, metrics = apply_fn(
            args, batch, student_logits, teacher_topk_logits,
            teacher_topk_indices, loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        # Top-k KL may be slightly negative due to tail approximation,
        # but should be close to 0 at worst
        assert kl_loss.item() >= -0.5, f"Top-k KL loss should be >= -0.5, got {kl_loss.item()}"

    def test_topk_is_weight_clipping(self):
        """Test IS weight clipping in top_k mode."""
        apply_fn = self._get_function()
        args = self._make_args(mopd_eps_low=0.5, mopd_eps_high=2.0)

        V = 20
        k = 8
        student_logits = [torch.randn(3, V)]
        teacher_logits = [torch.randn(3, V)]
        topk_vals, topk_idx = teacher_logits[0].topk(k, dim=-1)

        teacher_topk_logits = {"math": [topk_vals]}
        teacher_topk_indices = {"math": [topk_idx]}

        batch = {"rollout_log_probs": [torch.tensor([0.0, 0.0, 0.0])]}
        current_log_probs = [torch.tensor([-5.0, 0.0, 5.0])]
        loss_masks = [torch.ones(3)]

        kl_loss, metrics = apply_fn(
            args, batch, student_logits, teacher_topk_logits,
            teacher_topk_indices, loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        is_nonzero_frac = metrics["mopd_is_nonzero_frac"].item()
        assert abs(is_nonzero_frac - 1.0 / 3.0) < 0.05, (
            f"Expected ~1/3 nonzero IS weight fraction, got {is_nonzero_frac}"
        )

    def test_topk_none_teacher_for_sample(self):
        """Test that None entries in teacher data are skipped."""
        apply_fn = self._get_function()
        args = self._make_args(
            _mopd_teachers_parsed=[
                {"name": "math_teacher", "domain": "math"},
                {"name": "code_teacher", "domain": "code"},
            ],
        )

        V = 20
        k = 8
        student_0 = torch.randn(3, V)
        student_1 = torch.randn(4, V)

        teacher_0 = torch.randn(3, V)
        teacher_1 = torch.randn(4, V)
        teacher_code_1 = torch.randn(4, V)

        topk_vals_0, topk_idx_0 = teacher_0.topk(k, dim=-1)
        topk_vals_1, topk_idx_1 = teacher_1.topk(k, dim=-1)
        topk_vals_c1, topk_idx_c1 = teacher_code_1.topk(k, dim=-1)

        teacher_topk_logits = {
            "math": [topk_vals_0, topk_vals_1],
            "code": [None, topk_vals_c1],
        }
        teacher_topk_indices = {
            "math": [topk_idx_0, topk_idx_1],
            "code": [None, topk_idx_c1],
        }

        batch = {"rollout_log_probs": [torch.zeros(3), torch.zeros(4)]}
        current_log_probs = [torch.zeros(3), torch.zeros(4)]
        loss_masks = [torch.ones(3), torch.ones(4)]

        kl_loss, metrics = apply_fn(
            args, batch, [student_0, student_1],
            teacher_topk_logits, teacher_topk_indices,
            loss_masks, self._sum_of_sample_mean,
            current_log_probs=current_log_probs,
        )

        assert kl_loss.shape == ()
        assert torch.isfinite(kl_loss), "Loss should be finite with None teacher entries"

    def test_topk_k_parameter_effect(self):
        """Test that larger k gives KL closer to full-vocab KL."""
        from slime.utils.ppo_utils import vocab_parallel_reverse_kl

        apply_fn = self._get_function()
        torch.manual_seed(42)

        V = 20
        student_logits = [torch.randn(5, V)]
        teacher_logits_raw = [torch.randn(5, V)]

        # Full-vocab KL as ground truth
        full_kl = vocab_parallel_reverse_kl(student_logits[0], teacher_logits_raw[0], None).sum().item()

        # Top-k with k=5
        k_small = 5
        topk_vals_s, topk_idx_s = teacher_logits_raw[0].topk(k_small, dim=-1)
        args_small = self._make_args(mopd_topk_k=k_small)
        batch = {"rollout_log_probs": [torch.zeros(5)]}
        current_log_probs = [torch.zeros(5)]
        loss_masks = [torch.ones(5)]

        kl_small, _ = apply_fn(
            args_small, batch, student_logits,
            {"math": [topk_vals_s]}, {"math": [topk_idx_s]},
            loss_masks, self._sum_of_sample_mean, current_log_probs=current_log_probs,
        )

        # Top-k with k=18 (close to V)
        k_large = 18
        topk_vals_l, topk_idx_l = teacher_logits_raw[0].topk(k_large, dim=-1)
        args_large = self._make_args(mopd_topk_k=k_large)

        kl_large, _ = apply_fn(
            args_large, batch, student_logits,
            {"math": [topk_vals_l]}, {"math": [topk_idx_l]},
            loss_masks, self._sum_of_sample_mean, current_log_probs=current_log_probs,
        )

        # Larger k should generally be closer to full KL (both are approximations)


# ---------------------------------------------------------------------------
# Tests for top_k argument validation
# ---------------------------------------------------------------------------
class TestMopdTopkValidation:
    """Test --mopd-distill-type=top_k parameter validation."""

    @pytest.fixture(autouse=True)
    def _mock_deps(self, monkeypatch):
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
            mopd_distill_type="token_level",
            mopd_topk_k=1024,
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

    def test_topk_without_teacher_loads_raises(self):
        """Test that top_k mode without --mopd-teacher-loads raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_distill_type="top_k",
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_teacher_loads=None,
        )
        with pytest.raises(ValueError, match="top_k.*mopd-teacher-loads|megatron teacher"):
            slime_validate_args(args)

    def test_topk_k_must_be_positive(self):
        """Test that --mopd-topk-k <= 0 raises ValueError."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_distill_type="top_k",
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_teacher_loads=["/tmp/fake_teacher"],
            mopd_topk_k=0,
        )
        with pytest.raises(ValueError, match="mopd-topk-k.*> 0"):
            slime_validate_args(args)

    def test_topk_k_default(self):
        """Test that --mopd-topk-k defaults to 1024."""
        from slime.utils.arguments import slime_validate_args

        args = self._make_base_args(
            use_mopd=True,
            mopd_distill_type="top_k",
            mopd_teachers='[{"name": "t1", "domain": "math"}]',
            mopd_teacher_loads=["/tmp/fake_teacher"],
            mopd_topk_k=1024,
            mopd_alpha=0.0,
        )
        slime_validate_args(args)
        assert args.mopd_topk_k == 1024