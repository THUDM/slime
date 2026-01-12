import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 4
CP_SIZE = 1
MEGATRON_TP_SIZE = 1
MEGATRON_PP_SIZE = 1
FORCE_GENERATE_TEST_DATA = False


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )
    
    debug_data_path = "test_rollout_data_0.pt"
    if not os.path.exists(debug_data_path) or FORCE_GENERATE_TEST_DATA:
        print(f"[Test] Generating test rollout data at {debug_data_path}...")
        generate_test_data(debug_data_path)
    else:
        print(f"[Test] Using existing test data at {debug_data_path}")
    
    return debug_data_path


def get_common_args() -> str:
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-mem-fraction-static 0.75 "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--use-fault-tolerance "
    )

    return (
        f"{rollout_args} "
        f"{ppo_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )


def generate_test_data(output_path: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_script = os.path.join(script_dir, "utils/test_data_generator.py")
    
    common_args = get_common_args()
    
    gen_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        "--train-backend fsdp "
        f"{common_args} "
        f"--save-debug-rollout-data {output_path} "
        "--debug-rollout-only "
    )
    
    U.execute_train(
        train_args=gen_args,
        train_script=gen_script,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
    )


def build_train_args(train_backend: str, debug_data_path: str, ci_grad_norm_mode: str = None) -> str:
    ref_load_path = f"/root/models/{MODEL_NAME}"
    if train_backend == "megatron":
        ref_load_path = f"/root/models/{MODEL_NAME}_torch_dist"

    ckpt_args = (
        f"--hf-checkpoint /root/models/{MODEL_NAME} "
        f"--ref-load {ref_load_path} "
    )

    common_args = get_common_args()

    debug_args = (
        f"--load-debug-rollout-data {debug_data_path} "
        "--debug-train-only "
    )

    if train_backend == "fsdp":
        train_backend_args = (
            "--train-backend fsdp "
            "--attn-implementation flash_attention_2 "
            "--gradient-checkpointing "
            f"--context-parallel-size {CP_SIZE} "
            f"--update-weight-buffer-size {512 * 1024 * 1024} "
            """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
        )
    else:
        train_backend_args = (
            f"--tensor-model-parallel-size {MEGATRON_TP_SIZE} "
            "--sequence-parallel "
            f"--pipeline-model-parallel-size {MEGATRON_PP_SIZE} "
            f"--context-parallel-size {CP_SIZE} "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--train-memory-margin-bytes 3221225472 "
        )

    ci_args = "--ci-test "
    grad_norm_path = "grad_norm_fsdp.pt"
    if ci_grad_norm_mode == "save":
        ci_args += f"--ci-save-grad-norm {grad_norm_path} "
    elif ci_grad_norm_mode == "load":
        ci_args += f"--ci-load-grad-norm {grad_norm_path} "

    return (
        f"{ckpt_args} "
        f"{common_args} "
        f"{debug_args} "
        f"{train_backend_args} "
        f"{ci_args} "
    )


def run_single_test(train_backend: str, debug_data_path: str, ci_grad_norm_mode: str = None):
    train_args = build_train_args(train_backend, debug_data_path, ci_grad_norm_mode)
    megatron_model_type = MODEL_TYPE if train_backend == "megatron" else None
    
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=megatron_model_type,
    )


def execute(debug_data_path: str):
    grad_norm_path = "grad_norm_fsdp.pt"
    
    try:
        print("[Test] Running FSDP and save grad norm...")
        run_single_test("fsdp", debug_data_path, ci_grad_norm_mode="save")
        
        print("[Test] Running Megatron and compare grad norm...")
        run_single_test("megatron", debug_data_path, ci_grad_norm_mode="load")

    finally:
        if os.path.exists(grad_norm_path):
            os.remove(grad_norm_path)


if __name__ == "__main__":
    debug_data_path = prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(debug_data_path)
