from dataclasses import dataclass
from typing import Literal

import typer

import slime.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-30B-A3B"
    megatron_model_type: str = "qwen3-30B-A3B"
    num_gpus_per_node: int | None = None
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    enable_eval: bool = True
    extra_args: str = ""
    rollout_fp8: bool = False
    rollout_attn_fp8: bool = False
    train_fp8: bool = False
    enable_megatron_bridge: bool = False
    enable_mis: bool = False
    # TODO improve, should be able to override more easily
    tis_use_rs: bool = True

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{args.model_name} --local-dir /root/models/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    if args.rollout_fp8:
        U.exec_command(
            f"huggingface-cli download Qwen/{args.model_name}-FP8 --local-dir /root/models/{args.model_name}-FP8"
        )

    if not args.enable_megatron_bridge:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst="/root/models",
        )


# TODO improve layering: split algorithm vs infra
def execute(args: ScriptArgs):
    ref_load_path = (
        f"/root/models/{args.model_name}/"
        if args.enable_megatron_bridge
        else f"/root/models/{args.model_name}_torch_dist"
    )
    load_save_path = f"/root/shared_data/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint /root/models/{args.model_name}{'-FP8' if args.rollout_fp8 else ''}/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
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
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details /root/shared_data/{args.run_id}/dump_details "
    )
    misc_env_vars = {}

    if args.train_fp8:
        match args.hardware:
            case "GB200" | "GB300":
                # It can run but accuracy is incorrect currently
                raise NotImplementedError
                # ref: Megatron-MoE-ModelZoo
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe mxfp8 "
                    "--fp8-param-gather "
                    "--reuse-grad-buf-for-mxfp8-param-ag "
                    # --moe-router-padding-for-quantization
                )
            case "H100" | "H200":
                # ref: fp8 blog
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe blockwise "
                    # "--fp8-param-gather "
                )
                misc_env_vars |= {
                    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
                }

    if args.enable_megatron_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

    match (args.hardware, args.num_nodes):
        case ("H100", 1):
            perf_args += (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                f"--rollout-num-gpus-per-engine {2 if args.rollout_fp8 else 8} "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-max-bs 512 "
            )
            optimizer_args += (
                "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
            )
        case ("GB200", 1) | ("GB300", 1) | ("GB200", 2) | ("GB300", 2) | ("GB200", 4) | ("GB300", 4):
            perf_args += (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 4 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                f"--rollout-num-gpus-per-engine {2 if args.rollout_fp8 else 4} "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-attention-backend trtllm_mha "
            )
            if args.rollout_fp8:
                sglang_world_size = 2
                sglang_attn_tp_size = 2
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--sglang-ep-size {sglang_world_size} "
                    "--sglang-moe-runner-backend deep_gemm "
                    "--sglang-moe-a2a-backend deepep "
                    f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
                )
            else:
                sglang_args += "--sglang-cuda-graph-max-bs 512 "
        case _:
            raise NotImplementedError

    if args.rollout_attn_fp8:
        sglang_args += "--sglang-kv-cache-dtype fp8_e4m3 "

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

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**misc_env_vars},
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
