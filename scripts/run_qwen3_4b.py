from dataclasses import dataclass
from typing import Literal, Optional

import typer

import slime.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: Optional[str] = None
    num_gpus_per_node: Optional[int] = None
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    extra_args: str = ""
    multi_eval: bool = False
    true_on_policy: bool = False
    dynamic_sampling: bool = False
    enable_eval: bool = True
    train_backend: Literal["fsdp", "megatron"] = "megatron"
    rollout_fp8: bool = False
    train_fp8: bool = False
    enable_megatron_bridge: bool = False
    enable_mis: bool = False
    # TODO improve, should be able to override more easily
    tis_use_rs: bool = True

    def __post_init__(self):
        if self.train_backend == "megatron":
            self.megatron_model_type = {
                "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
                "Qwen3-4B-Base": "qwen3-4B",
                "Qwen3-4B": "qwen3-4B",
            }[self.model_name]

        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{args.model_name} --local-dir /root/models/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    if args.multi_eval:
        U.hf_download_dataset("zyzshishui0627/gpqa_diamond")
        U.hf_download_dataset("zyzshishui0627/IFBench")

    if args.rollout_fp8:
        U.exec_command(
            f"huggingface-cli download Qwen/{args.model_name}-FP8 --local-dir /root/models/{args.model_name}-FP8"
        )

    if (args.train_backend == "megatron") and not args.enable_megatron_bridge:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst="/root/models",
        )


def execute(args: ScriptArgs):
    load_save_path = f"/root/shared_data/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint /root/models/{args.model_name}{'-FP8' if args.rollout_fp8 else ''} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )
    if args.train_backend == "megatron":
        ref_load_path = (
            f"/root/models/{args.model_name}/"
            if args.enable_megatron_bridge
            else f"/root/models/{args.model_name}_torch_dist"
        )
        ckpt_args += (
            # FSDP does not support this
            f"--ref-load {ref_load_path} "
        )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        # By default it is thinking mode
        # """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 0.8 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    if args.dynamic_sampling and (args.true_on_policy != "debug_minimal"):
        rollout_args += (
            "--over-sampling-batch-size 64 "
            "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    # sometimes disable eval to speed up debugging
    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_max_response_len = 16384
        eval_args += "--eval-interval 20 "
        if args.multi_eval:
            eval_config_text = f"""
eval:
  defaults:
    max_response_len: {eval_max_response_len}
    top_p: 0.7
  datasets:
    - name: aime
      path: /root/datasets/aime-2024/aime-2024.jsonl
      rm_type: math
      n_samples_per_eval_prompt: 16
    - name: gpqa
      path: /root/datasets/gpqa_diamond/gpqa_eval.jsonl
      rm_type: gpqa
      n_samples_per_eval_prompt: 2
    - name: ifbench
      path: /root/datasets/IFBench/IFBench_eval.jsonl
      rm_type: ifbench
      n_samples_per_eval_prompt: 1
""".strip()
            eval_args += f"--eval-config {U.save_to_temp_file(eval_config_text, 'yaml')} "
        else:
            eval_args += (
                "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 16 "
                f"--eval-max-response-len {eval_max_response_len} "
                "--eval-top-p 0.7 "
            )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        # "--fsdp-cpu-offload "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = f"--rollout-num-gpus-per-engine 1 " "--sglang-chunked-prefill-size 4096 "

    match args.train_backend:
        case "fsdp":
            train_backend_args = (
                "--train-backend fsdp "
                "--attn-implementation flash_attention_2 "
                "--gradient-checkpointing "
                f"--update-weight-buffer-size {512 * 1024 * 1024} "  # 512MB
                """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
            )
            sglang_args += f"--sglang-mem-fraction-static 0.75 "
            perf_args = "--use-dynamic-batch-size " "--max-tokens-per-gpu 32768 "

        case "megatron":
            train_backend_args = (
                f"--tensor-model-parallel-size {2 if args.num_gpus_per_node == 8 else 1} "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                f"--context-parallel-size {4 if args.num_gpus_per_node == 8 else 1} "
                "--expert-model-parallel-size 1 "
                "--expert-tensor-parallel-size 1 "
                "--recompute-granularity full "
                "--recompute-method uniform "
                "--recompute-num-layers 1 "
                # default dropout in megatron is 0.1
                "--attention-dropout 0.0 "
                "--hidden-dropout 0.0 "
                # should be good for model performance
                "--accumulate-allreduce-grads-in-fp32 "
                "--attention-softmax-in-fp32 "
                # need to comment this when using model with MLA
                "--attention-backend flash "
                "--train-memory-margin-bytes 3221225472 "
            )
            # TODO improve
            sglang_args += f"--sglang-mem-fraction-static 0.7 "
            perf_args = "--use-dynamic-batch-size " "--max-tokens-per-gpu 9216 "

        case _:
            raise NotImplementedError

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details /root/shared_data/{args.run_id}/dump_details "
    )
    misc_env_vars = {}

    if args.model_name == "Qwen3-4B-Base":
        misc_args += "--sglang-context-length 36000 "
        misc_env_vars |= {
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }

    if args.train_fp8:
        misc_args += (
            "--transformer-impl transformer_engine "
            "--bf16 "
            "--fp8-format e4m3 "
            "--fp8-recipe blockwise "
            "--fp8-param-gather "
        )
        misc_env_vars |= {
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        }

    if args.enable_megatron_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

    if args.enable_mis:
        config_text = f"""
use_tis: true
use_rs: {"true" if args.tis_use_rs else "false"}
tis_level: "token"
rs_level: "token"
tis_mode: "truncate"
tis_lower_bound: 0.5
tis_upper_bound: 2.0
rs_lower_bound: null
rs_upper_bound: null
rs_veto_threshold: 1.0e-4
tis_batch_normalize: true
""".strip()
        misc_args += (
            f"--custom-config-path {U.save_to_temp_file(config_text, 'yaml')} "
            "--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    true_on_policy_args = ""
    true_on_policy_envs = {}
    if args.true_on_policy:
        true_on_policy_args = (
            "--sglang-enable-deterministic-inference "
            "--sglang-rl-on-policy-target fsdp "
            "--sglang-attention-backend fa3 "
            "--attn-implementation flash_attention_3 "
            "--deterministic-mode "
            "--true-on-policy-mode "
        )
        true_on_policy_envs = {
            "NCCL_ALGO": "allreduce:tree",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        # TODO may get it from `config`
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            **misc_env_vars,
            **true_on_policy_envs,
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
