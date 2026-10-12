r"""Real Prism native-load, selected-action backward and MCore Adam smoke.

Run inside the GPU image with Megatron on PYTHONPATH, for example::

    PYTHONPATH=/opt/megatron:$PWD torchrun --standalone --nproc-per-node=1 \
      scripts/test_prism_gpu.py \
      --hf-checkpoint /shared/prism-hf \
      --prism-pretrained-checkpoint /shared/prism-native \
      --artifact-dir /shared/prism-smoke \
      --numpy-overlay /shared/numpy1 \
      --omni-url http://worker-or-router:port \
      --prompt-json /shared/prism-prompt.json --alignment-atol 0.05

The tolerance above is an example: choose it explicitly for the tested dtype
and kernels. Both RVQ and stop differences are reported before any update.
Without an Omni URL or saved /generate response this runs a synthetic carrier
smoke, which verifies GPU execution but does not establish engine alignment.
NCCL publication can be added with --after-update-hook module.function. The
hook receives the still-live CUDA student, optimizer, replay and score results.
"""

import argparse
import json
import math
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path


def cli_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-checkpoint", required=True, help="Converted Prism inference artifact directory.")
    parser.add_argument(
        "--prism-pretrained-checkpoint", required=True, help="Native RUN/checkpoint/iteration directory."
    )
    parser.add_argument("--prism-pretrained-step", type=int)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--numpy-overlay", help="Directory containing NumPy 1.x, prepended before importing Megatron.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--omni-url", help="Live Omni worker/router URL; /generate is appended if needed.")
    source.add_argument("--trajectory-json", help="Saved complete Omni /generate response JSON.")
    parser.add_argument("--prompt-json", help="Original structured Omni prompt JSON, required for real trajectories.")
    parser.add_argument("--omni-model-name", default="student")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument(
        "--alignment-atol", type=float, help="Required explicit numerical bound for Omni trajectories."
    )
    parser.add_argument("--alignment-rtol", type=float, default=0.0)
    parser.add_argument(
        "--expected-weight-version", help="Optional required admission version for the real trajectory."
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--before-trajectory-hook", help="Optional function(context) to publish initial weights before generation."
    )
    parser.add_argument("--after-update-hook", help="Optional function(context) for publication/serving verification.")
    values, megatron_flags = parser.parse_known_args()
    if (values.omni_url or values.trajectory_json) and not values.prompt_json:
        parser.error("Omni replay requires --prompt-json; task/reference metadata must be explicit")
    if (values.omni_url or values.trajectory_json) and values.alignment_atol is None:
        parser.error("Omni replay requires an explicit --alignment-atol")
    for name in ("alignment_atol", "alignment_rtol", "learning_rate", "stop_loss_weight"):
        value = getattr(values, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if values.learning_rate == 0 or values.max_frames < 1:
        parser.error("learning rate and max frames must be positive")
    return values, megatron_flags


def setup(args, extra_flags):
    if args.numpy_overlay:
        sys.path.insert(0, str(Path(args.numpy_overlay).resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.core import mpu, tensor_parallel
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_args

    from slime_plugins.models.prism import configure_prism_args

    if not np.__version__.startswith("1."):
        raise RuntimeError(f"Megatron requires NumPy 1.x; found {np.__version__}. Use --numpy-overlay.")
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires a real CUDA GPU and Transformer Engine.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This first GPU smoke supports one learner rank, TP=PP=CP=DP=1.")
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *extra_flags]
        runtime = parse_args()
    finally:
        sys.argv = previous_argv
    runtime.hf_checkpoint = str(Path(args.hf_checkpoint).resolve())
    runtime.prism_pretrained_checkpoint = str(Path(args.prism_pretrained_checkpoint).resolve())
    runtime.prism_pretrained_step = args.prism_pretrained_step
    spec = configure_prism_args(runtime)
    # A smoke creates its own fresh RL optimizer. Architecture/dtype comes
    # exclusively from the inference artifact validated against native DCP.
    runtime.bf16, runtime.fp16 = spec.dtype == "bfloat16", spec.dtype == "float16"
    runtime.micro_batch_size = 1
    runtime.global_batch_size = 1
    runtime.seq_length = max(128, runtime.seq_length or 128)
    runtime.train_iters = 1
    runtime.lr = args.learning_rate
    runtime.hidden_dropout = runtime.attention_dropout = 0.0
    runtime.variable_seq_lengths = True
    runtime.gradient_accumulation_fusion = False
    validate_args(runtime)
    if (runtime.tensor_model_parallel_size, runtime.pipeline_model_parallel_size, runtime.context_parallel_size) != (
        1,
        1,
        1,
    ):
        raise ValueError("This first GPU smoke requires TP=PP=CP=1.")
    set_args(runtime)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if "MASTER_ADDR" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(seconds=args.request_timeout))
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=args.request_timeout),
        )
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        create_gloo_process_groups=False,
    )
    torch.manual_seed(runtime.seed)
    tensor_parallel.model_parallel_cuda_manual_seed(runtime.seed)
    return runtime, spec, np.__version__


def input_samples(args, spec, output):
    import torch

    from slime.utils.types import Sample
    from slime_plugins.omni_rollout import PrismConfig, build_generate_payload, parse_rollout

    if args.omni_url or args.trajectory_json:
        sample = Sample(index=0, group_index=0)
        sample.metadata = {"omni_prompt": json.loads(Path(args.prompt_json).read_text())}
        sample.prompt = sample.metadata["omni_prompt"]["script"]
        if args.omni_url:
            import urllib.request

            args.omni_return_audio = False
            payload = build_generate_payload(args, sample, {"max_new_tokens": args.max_frames, "seed": 42})
            url = args.omni_url.rstrip("/")
            if not url.endswith("/generate"):
                url += "/generate"
            (output / "generate-request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            request = urllib.request.Request(
                url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                result = json.load(response)
        else:
            result = json.loads(Path(args.trajectory_json).read_text())
        (output / "generate-response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        parse_rollout(
            result,
            sample,
            PrismConfig.from_path(args.hf_checkpoint),
            expected_weight_version=args.expected_weight_version,
        )
        return [sample], "omni"

    samples = []
    for index, (frames, natural, advantage) in enumerate(((2, True, 1.0), (2, False, -0.6), (0, True, 0.4))):
        q, rows = spec.topology.n_vq, frames + 1
        codes = (torch.arange(frames * q).reshape(frames, q) % max(spec.speech_vocab_size - 1, 1)) + 1
        codes %= spec.speech_vocab_size
        mask = torch.arange(rows) < frames
        stops = mask.clone()
        stop_actions = torch.zeros(rows, dtype=torch.long)
        if natural:
            stops[-1], stop_actions[-1] = True, 1
        retention = torch.ones(rows, q, dtype=torch.bool)
        retention[1:] = False
        retention[1:, torch.tensor(spec.input_rvq_channels) - 1] = True
        sample = Sample(
            index=index,
            group_index=0,
            tokens=[spec.audio_start_token_id, *([spec.text_pad_idx] * frames)],
            response_length=frames,
            loss_mask=[1] * frames,
            reward=advantage,
        )
        sample.multimodal_train_inputs = {
            "audio_codes": torch.cat((torch.zeros(1, q, dtype=torch.long), codes)),
            "item_kind": torch.tensor([0, *([1] * frames)]),
            "audio_role": torch.tensor([0, *([2] * frames)]),
            "successor_audio_codes": torch.cat((codes, torch.zeros(1, q, dtype=torch.long))),
            "successor_audio_mask": mask,
            "target_history_retention_mask": retention,
            "policy_code_mask": mask[:, None].expand(rows, q).clone(),
            "policy_stop_mask": stops,
            "policy_stop_actions": stop_actions,
            "policy_group_advantage": torch.tensor([advantage]),
        }
        samples.append(sample)
    return samples, "synthetic"


def policy_args(runtime, args):
    values = vars(runtime).copy()
    values.update(
        custom_megatron_log_prob_function_path="slime_plugins.models.prism.extract_policy_log_probs",
        policy_stop_loss_weight=args.stop_loss_weight,
        advantage_estimator="grpo",
        loss_type="policy_loss",
        use_rollout_logprobs=False,
        use_opd=False,
        normalize_advantages=False,
        kl_coef=0.0,
        use_opsm=False,
        use_tis=False,
        get_mismatch_metrics=False,
        use_score_centering=False,
        use_kl_loss=False,
        custom_pg_loss_reducer_function_path=None,
        custom_advantage_function_path=None,
        compute_advantages_and_returns=True,
        allgather_cp=False,
        calculate_per_token_loss=False,
        pg_loss_type="ppo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        entropy_coef=0.0,
        use_rollout_entropy=False,
        rollout_top_p=1.0,
        recompute_loss_function=False,
    )
    return argparse.Namespace(**values)


def make_batch(samples):
    import torch

    from slime.backends.megatron_utils.data import DataIterator, get_batch

    device = torch.device("cuda", torch.cuda.current_device())
    mm = [{key: value.to(device) for key, value in sample.multimodal_train_inputs.items()} for sample in samples]
    rollout = {
        "tokens": [torch.tensor(sample.tokens, dtype=torch.long, device=device) for sample in samples],
        "loss_masks": [torch.tensor(sample.loss_mask, device=device) for sample in samples],
        "total_lengths": [len(sample.tokens) for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "multimodal_train_inputs": mm,
        # The test fixture supplies a nonzero advantage to exercise gradients;
        # this is not a reward-model or complete-prompt-group correctness test.
        "rewards": [sample.reward if sample.reward is not None else 1.0 for sample in samples],
    }
    iterator = DataIterator(rollout, [list(range(len(samples)))])
    batch = get_batch(iterator, list(rollout), pad_multiplier=128, allgather_cp=False)
    batch["rollout_mask_sums"] = torch.tensor([max(sample.response_length, 1) for sample in samples], device=device)
    return rollout, batch


def forward(model, batch):
    return model(
        input_ids=batch["tokens"],
        position_ids=None,
        attention_mask=None,
        packed_seq_params=batch["packed_seq_params"],
        **batch["multimodal_train_inputs"],
    )


def compare_omni_scores(args, samples, scores):
    import torch

    diagnostics = {}
    for kind in ("code", "stop"):
        actual, expected = [], []
        for sample, values in zip(samples, scores[f"policy_{kind}_logprobs"], strict=True):
            mm = sample.multimodal_train_inputs
            mask = mm[f"policy_{kind}_mask"].to(values.device)
            actual.append(values[mask].detach().float().cpu())
            expected.append(mm[f"policy_rollout_{kind}_logprobs"][mm[f"policy_{kind}_mask"]].float().cpu())
        actual, expected = torch.cat(actual), torch.cat(expected)
        delta = (actual - expected).abs()
        diagnostics[kind] = {
            "count": delta.numel(),
            "max_abs_diff": delta.max().item() if delta.numel() else None,
            "mean_abs_diff": delta.mean().item() if delta.numel() else None,
            "aligned": bool(torch.allclose(actual, expected, atol=args.alignment_atol, rtol=args.alignment_rtol)),
        }
    return diagnostics


def cpu_values(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: cpu_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_values(item) for item in value]
    return value


def run(args, extra_flags):
    if args.numpy_overlay:
        sys.path.insert(0, str(Path(args.numpy_overlay).resolve()))
    import torch
    from megatron.core import tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from slime.backends.megatron_utils.loss import compute_advantages_and_returns, policy_loss_function
    from slime.backends.megatron_utils.policy_actions import store_old_action_scores
    from slime_plugins.models.prism import extract_policy_log_probs, load_prism_pretrained, model_provider

    runtime, spec, numpy_version = setup(args, extra_flags)
    output = Path(args.artifact_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "hf_checkpoint": runtime.hf_checkpoint,
        "native_checkpoint": runtime.prism_pretrained_checkpoint,
        "numpy": numpy_version,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "steps": 0,
        "model": {"dtype": spec.dtype, "physical_layers": spec.topology.num_layers, "n_vq": spec.topology.n_vq},
    }
    try:
        student = model_provider().to(device="cuda", dtype=getattr(torch, spec.dtype))
        for parameter in student.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
        if not all(parameter.requires_grad for parameter in student.parameters()):
            raise AssertionError("Prism smoke must leave every parameter trainable")
        model = DistributedDataParallel(
            student.config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=False, check_for_nan_in_grad=True
            ),
            student,
        )
        load_prism_pretrained([model], runtime)
        report["native_loaded"] = True
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=args.learning_rate,
                min_lr=0.0,
                weight_decay=0.0,
                bf16=runtime.bf16,
                fp16=runtime.fp16,
                params_dtype=getattr(torch, spec.dtype),
                clip_grad=1.0,
            ),
            [model],
            use_gloo_process_groups=False,
        )
        learner_args = policy_args(runtime, args)
        context = dict(
            args=args,
            learner_args=learner_args,
            runtime=runtime,
            model=model,
            optimizer=optimizer,
            report=report,
            artifact_dir=output,
        )
        if args.before_trajectory_hook:
            from slime.utils.misc import load_function

            report["before_trajectory"] = load_function(args.before_trajectory_hook)(context)
        samples, report["trajectory_source"] = input_samples(args, spec, output)
        rollout, batch = make_batch(samples)
        report["replay_action_counts"] = {
            "frames": sum(sample.response_length for sample in samples),
            "rvq": int(batch["multimodal_train_inputs"]["policy_code_mask"].sum()),
            "stop": int(batch["multimodal_train_inputs"]["policy_stop_mask"].sum()),
        }
        model.eval()
        with torch.no_grad():
            before = extract_policy_log_probs(learner_args, batch, forward(model, batch))
        if report["trajectory_source"] == "omni":
            report["alignment"] = compare_omni_scores(args, samples, before)
            (output / "alignment.json").write_text(json.dumps(report["alignment"], indent=2))
            if not all(value["aligned"] for value in report["alignment"].values()):
                raise AssertionError("Pre-update Omni/Megatron probabilities exceed the requested alignment tolerance")
        if not any(sample.response_length for sample in samples):
            raise AssertionError(
                "All trajectories stopped immediately; use another prompt/seed to exercise RVQ gradients"
            )
        store_old_action_scores(rollout, before)
        rollout["log_probs"] = before["log_probs"]
        compute_advantages_and_returns(learner_args, rollout)
        # Repack after attaching frozen old scores and scalar advantages.
        from slime.backends.megatron_utils.data import DataIterator, get_batch

        batch = get_batch(DataIterator(rollout, [list(range(len(samples)))]), list(rollout), 128, False)
        batch["rollout_mask_sums"] = torch.tensor([max(s.response_length, 1) for s in samples], device="cuda")
        (output / "replay.json").write_text(
            json.dumps(
                cpu_values(
                    {
                        "tokens": [s.tokens for s in samples],
                        "response_lengths": rollout["response_lengths"],
                        "total_lengths": rollout["total_lengths"],
                        "multimodal_train_inputs": rollout["multimodal_train_inputs"],
                    }
                ),
                indent=2,
            )
        )
        # Snapshot only one short, actually selected row per head, plus stop.
        # This is an update assertion, not a whole-model hash or duplicated weights.
        probe_params = {f"rvq_{i}": head.weight for i, head in enumerate(student.audio_lm_heads)}
        probe_params["stop"] = student.stop_head.weight
        mm = batch["multimodal_train_inputs"]
        probe_rows = {
            f"rvq_{i}": int(mm["successor_audio_codes"][mm["policy_code_mask"][:, i], i][0])
            for i in range(spec.topology.n_vq)
        }
        probe_rows["stop"] = 0
        probes = {
            name: parameter[probe_rows[name], :64].detach().float().clone() for name, parameter in probe_params.items()
        }
        model.train()
        model.zero_grad_buffer()
        optimizer.zero_grad()
        logits = forward(model, batch)
        value, metrics = policy_loss_function(learner_args, batch, logits, lambda values: values.sum())
        value = value / len(samples)
        if not torch.isfinite(value):
            raise AssertionError("Policy objective is not finite")
        optimizer.scale_loss(value).backward()
        finalize_model_grads([model])
        grads = {}
        for name, parameter in probe_params.items():
            gradient = getattr(parameter, "main_grad", parameter.grad)
            if gradient is None or not torch.isfinite(gradient).all():
                raise AssertionError(f"Missing/nonfinite gradient in {name}")
            grads[name] = gradient.abs().max().item()
            if grads[name] == 0:
                raise AssertionError(f"No selected-action gradient in {name}")
        success, grad_norm, _ = optimizer.step()
        if not success:
            raise AssertionError("MCore optimizer skipped the update")
        changes = {
            name: (parameter[probe_rows[name], :64].detach().float() - probes[name]).abs().max().item()
            for name, parameter in probe_params.items()
        }
        if any(change == 0 for change in changes.values()):
            raise AssertionError(f"A policy head did not change in stored precision: {changes}")
        model.eval()
        with torch.no_grad():
            after = extract_policy_log_probs(learner_args, batch, forward(model, batch))
        report.update(
            status="passed",
            steps=1,
            loss=value.item(),
            gradient_norm=float(grad_norm),
            head_gradient_absmax=grads,
            head_parameter_delta_absmax=changes,
            loss_metrics=cpu_values(metrics),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        torch.save(
            {
                "before": {key: [value.detach().cpu() for value in values] for key, values in before.items()},
                "after": {key: [value.detach().cpu() for value in values] for key, values in after.items()},
            },
            output / "scores.pt",
        )
        context.update(
            samples=samples,
            rollout=rollout,
            batch=batch,
            before_scores=before,
            after_scores=after,
            report=report,
            artifact_dir=output,
        )
        if args.after_update_hook:
            from slime.utils.misc import load_function

            report["after_update"] = load_function(args.after_update_hook)(context)
        return context
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(cpu_values(report), indent=2))
        print(json.dumps(cpu_values(report), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    cli, remaining = cli_args()
    if cli.numpy_overlay:
        sys.path.insert(0, str(Path(cli.numpy_overlay).resolve()))
    try:
        run(cli, remaining)
    finally:
        # Destroy only groups established by this single-rank smoke.
        import torch.distributed as dist

        if dist.is_initialized():
            from megatron.core import mpu

            mpu.destroy_model_parallel()
            dist.destroy_process_group()
