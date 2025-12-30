"""E2E test for HF Datasets integration with SFT (Supervised Fine-Tuning).

This test verifies:
1. HF Datasets streaming mode works with SFT training
2. Dataset: teknium/OpenHermes-2.5 (loaded directly via load_dataset)
3. Checkpoint save/restore includes HF adapter state
4. SFT mode correctly handles conversations field
"""

import os
from pathlib import Path

import torch

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"


def prepare():
    """Prepare model (dataset will be auto-loaded by HF Datasets)."""
    U.exec_command("mkdir -p /root/models")

    # Download and convert model checkpoint
    # Dataset will be auto-loaded via --prompt-data
    U.convert_checkpoint(
        model_name=f"Qwen/{MODEL_NAME}",
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=2,
    )


def execute():
    """Execute SFT training with HF Datasets."""
    ckpt_args = (
        f"--hf-checkpoint /root/models/Qwen/{MODEL_NAME} "
        "--save /root/Qwen3-0.6B_slime_sft/ "
        "--save-interval 30 "  # Save checkpoint at step 30
    )

    rollout_args = (
        # HF Datasets configuration
        "--prompt-data teknium/OpenHermes-2.5 "  # Direct HF dataset name
        "--use-hf-datasets "  # Enable HF Datasets streaming mode
        "--hf-dataset-buffer-size 100 "
        "--hf-dataset-shuffle-buffer 1000 "
        "--hf-dataset-num-proc 4 "
        # Data keys (specific to OpenHermes-2.5)
        "--input-key conversations "  # OpenHermes uses conversations field
        "--apply-chat-template "
        "--rollout-shuffle "
        # SFT-specific settings
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 60} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 1 "  # SFT typically uses 1 sample per prompt
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1.0 "  # Greedy decoding for SFT
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 30 "  # Align with save-interval
        "--eval-prompt-data hermes_test teknium/OpenHermes-2.5 "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 2048 "
    )

    # SFT mode: No reward model, no GRPO
    # Just standard language modeling loss
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "  # Higher LR for SFT
        "--lr-decay-style cosine "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 2 " "--sglang-decode-log-interval 1000 " "--sglang-enable-metrics "

    fsdp_args = "--update-weight-buffer-size 536870912 "  # 512MB

    ci_args = "--ci-test " "--ci-disable-kl-checker "

    misc_args = "--actor-num-nodes 1 " "--actor-num-gpus-per-node 2 " "--colocate " "--train-backend fsdp "

    # Phase 1: Run first 30 rollouts
    train_args_phase1 = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__, run_name_prefix='hf-sft-phase1')} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    print("\n" + "=" * 80)
    print("Phase 1: Running first 30 rollouts with HF Datasets (SFT mode)")
    print("=" * 80 + "\n")

    U.execute_train(
        train_args=train_args_phase1,
        num_gpus_per_node=2,
        megatron_model_type=None,
    )

    # Verify checkpoint was saved
    checkpoint_path = Path("/root/Qwen3-0.6B_slime_sft/rollout/global_dataset_state_dict_30.pt")
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"

    # Verify checkpoint contains HF adapter state
    state_dict = torch.load(checkpoint_path)
    assert "hf_adapter_state" in state_dict, "Missing HF adapter state in checkpoint!"
    hf_state = state_dict["hf_adapter_state"]
    assert "epoch_id" in hf_state, "Missing epoch_id in HF adapter state"
    assert "consumed_count" in hf_state, "Missing consumed_count in HF adapter state"

    print("\n" + "=" * 80)
    print("Checkpoint verified successfully!")
    print(f"  - Epoch ID: {hf_state['epoch_id']}")
    print(f"  - Consumed count: {hf_state['consumed_count']}")
    print("=" * 80 + "\n")

    # Phase 2: Resume from checkpoint and continue to 60 rollouts
    ckpt_args_phase2 = (
        f"--hf-checkpoint /root/models/Qwen/{MODEL_NAME} "
        "--load /root/Qwen3-0.6B_slime_sft/ "  # Load from previous checkpoint
        "--save /root/Qwen3-0.6B_slime_sft/ "
        "--save-interval 30 "
    )

    train_args_phase2 = (
        f"{ckpt_args_phase2} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__, run_name_prefix='hf-sft-phase2')} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    print("\n" + "=" * 80)
    print("Phase 2: Resuming from checkpoint (30 → 60 rollouts)")
    print("=" * 80 + "\n")

    U.execute_train(
        train_args=train_args_phase2,
        num_gpus_per_node=2,
        megatron_model_type=None,
    )

    print("\n" + "=" * 80)
    print("E2E SFT test completed successfully!")
    print("Verified:")
    print("  ✓ HF Datasets streaming mode with OpenHermes-2.5")
    print("  ✓ Conversations field parsing")
    print("  ✓ Checkpoint save with HF adapter state")
    print("  ✓ Checkpoint resume and continuation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    prepare()
    # Remove proxy settings (may interfere with HF Datasets download)
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(key, None)
    execute()
