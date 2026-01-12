"""
This file is in preview, and will be further refined and optimized.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import slime.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-4.5"
    megatron_model_type: str = "glm4.5-355B-A32B"
    num_gpus_per_node: int = 4
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    enable_eval: bool = True
    extra_args: str = ""
    rollout_fp8: bool = False
    rollout_attn_fp8: bool = False
    enable_mtp: bool = False  # TODO enable by default
    dynamic_sampling: bool = False
    enable_benchmark: bool = False
    enable_mis: bool = False
    # TODO improve, should be able to override more easily
    tis_use_rs: bool = True
    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    """This script only needs to be executed on one node."""
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(
        f"huggingface-cli download {args.model_org}/{args.model_name} --local-dir /root/models/{args.model_name}"
    )
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k")
            U.hf_download_dataset("zhuzilin/aime-2024")
            U.hf_download_dataset("zhuzilin/aime-2025")
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k")

    if args.rollout_fp8:
        _convert_hf_to_fp8(args)


def _convert_hf_to_fp8(args: ScriptArgs):
    path_output = f"/root/models/{args.model_name}-FP8/"
    if Path(path_output).exists():
        return

    U.exec_command(
        "python tools/convert_hf_to_fp8.py "
        f"--model-dir /root/models/{args.model_name} "
        f"--save-dir {path_output} "
        "--strategy block --block-size 128 128 "
        "--max-workers 4"
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=True,
        dir_dst="/root/models",
    )


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    _prepare_cp(args)


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(
        path_src=f"/root/models/{args.model_name}_torch_dist",
        path_dst=f"/root/local_data/{args.model_name}_torch_dist",
    )
    U.rsync_simple(
        path_src=f"/root/models/{args.model_name}",
        path_dst=f"/root/local_data/{args.model_name}",
    )
    if args.rollout_fp8:
        U.rsync_simple(
            path_src=f"/root/models/{args.model_name}-FP8",
            path_dst=f"/root/local_data/{args.model_name}-FP8",
        )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    # ensure files are there is it was not synced before
    _prepare_cp(args)

    assert args.hardware != "H100", "H100 is not yet supported in this script"

    hf_checkpoint = (
        f"/root/local_data/{args.model_name}-FP8" if args.rollout_fp8 else f"/root/local_data/{args.model_name}"
    )

    load_save_path = f"/root/shared_data/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load /root/local_data/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 10} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 10} "
    )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        # TODO enlarge
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 1 "
        # ------------
        # TODO enlarge
        "--num-steps-per-rollout 1 "
        "--balance-data "
        "--rollout-stop-token-ids 151329 151336 151338 "
    )

    if args.dynamic_sampling and (args.true_on_policy != "debug_minimal"):
        rollout_args += (
            "--over-sampling-batch-size 256 "
            "--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    # sometimes disable eval to speed up debugging
    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 1 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
            )
            eval_args += (
                "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 8192 "
            )
        case "gsm8k":
            rollout_args += (
                "--prompt-data /root/datasets/gsm8k/train.parquet "
                "--input-key messages "
                # Deliberately make it very short for this easy task
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    if args.num_nodes <= 4:
        # Not really runnable, useful for --debug-rollout-only
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        perf_args = (
            # TODO choose a good config
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            f"--pipeline-model-parallel-size {8 if args.num_nodes == 8 else 4} "
            "--context-parallel-size 2 "
            "--expert-model-parallel-size 8 "
            "--expert-tensor-parallel-size 1 "
        )
    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # TODO enables use-kl-loss but w/ coef 0. can we just disable it like this?
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        # TODO wrong?
        "--eps-clip 1e-4 "
        "--eps-clip-high 2e-4 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # ------------
        # "--optimizer-cpu-offload "
        # "--overlap-cpu-optimizer-d2h-h2d "
        # "--use-precision-aware-optimizer "
    )

    sglang_world_size = min(32, args.num_gpus_per_node * args.num_nodes)
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        # TODO improve
        f"--sglang-mem-fraction-static {0.8 if args.hardware == 'GB300' else 0.7} "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * 2048} "
        # make every dp rank has 128 concurrency
        # "--sglang-server-concurrency 1024 "
        # For quick experiments
        # """--sglang-json-model-override-args '{"num_hidden_layers": 5}' """
    )
    sglang_extra_env_vars = {}
    if U.GENERATION_HARDWARE[args.hardware] == "Blackwell":
        sglang_args += "--sglang-attention-backend trtllm_mha "
    if args.rollout_fp8:
        sglang_decode_max_bs = 256
        sglang_attn_tp_size = min(8, sglang_world_size)
        sglang_attn_dp_size = sglang_world_size // sglang_attn_tp_size
        sglang_args += (
            f"--sglang-ep-size {sglang_world_size} "
            "--sglang-enable-dp-attention "
            f"--sglang-dp-size {sglang_attn_dp_size} "
            "--sglang-moe-dense-tp-size 1 "
            "--sglang-enable-dp-lm-head "
            "--sglang-moe-runner-backend deep_gemm "
            "--sglang-moe-a2a-backend deepep "
            "--sglang-deepep-mode low_latency "
            f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
            f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
            f"--sglang-cuda-graph-max-bs {sglang_decode_max_bs} "
        )
        sglang_extra_env_vars |= {
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": f"{sglang_decode_max_bs}",
        }
    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 1 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 2 "
            "--sglang-enable-draft-weights-cpu-backup "
        )
        sglang_extra_env_vars |= {
            "SGLANG_ENABLE_SPEC_V2": "1",
        }

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        # 4GB will lead to oom, not checked yet
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        # TODO maybe enable it
        # use deepep for megatron
        # "--moe-enable-deepep "
        # "--moe-token-dispatcher-type flex "
        # ------------
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details /root/shared_data/{args.run_id}/dump_details "
        "--disable-weights-backuper "
        # TODO if good, also configure to other scripts
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
    )

    if args.enable_benchmark:
        misc_args += (
            "--custom-generate-function-path slime.rollout.generate_hub.benchmarkers.generate_with_random_osl "
            "--rollout-batch-size 128 "
            "--n-samples-per-prompt 8 "
            "--use-distributed-post "
            "--router-policy round_robin "
            "--sglang-server-concurrency 10000 "
            # GB200 w/ mem-frac 0.8 will lead to oom in long jobs currently, but here we use large value to make baseline more fair
            f"--sglang-mem-fraction-static {0.8 if args.hardware == 'GB300' else 0.75} "
        )

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
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            **sglang_extra_env_vars,
            # TODO handle these
            # "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
            # "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
            # "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
            # "NCCL_IB_TC": "160",
            # "NCCL_PXN_DISABLE": "0",
            # "NCCL_IB_GID_INDEX": "3",
            # "NCCL_NET_GDR_LEVEL": "4",
            # "NCCL_IB_RETRY_CNT": "7",
            # "NCCL_IB_TIMEOUT": "32",
            # "NCCL_IB_QPS_PER_CONNECTION": "8",
            # "NCCL_P2P_LEVEL": "NVL",
            # "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # TODO should this be used
            # "NCCL_NVLS_ENABLE": "0",
            # "NCCL_MIN_CTAS": "4",
            # "OMPI_MCA_pml": "ob1",
            # "OMPI_MCA_btl": "^openib",
            # "OMPI_MCA_routed": "direct",
            # "OMPI_MCA_routed_radix": "1024",
            # "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
            # "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
            # "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}",
        },
    )


if __name__ == "__main__":
    app()
