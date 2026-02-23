import os
import subprocess
import time
import urllib.request

import slime.utils.external_utils.command_utils as U

TIGHT_DEVICE_MEMORY = U.get_bool_env_var("SLIME_TEST_TIGHT_DEVICE_MEMORY", "1")

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
NUM_TRAIN_GPUS = 4

TEACHER_HOST = "127.0.0.1"
TEACHER_PORT = 13141


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"huggingface-cli download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _get_gpu_split():
    """Split available GPUs: first half for training, one from the rest for teacher."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(NUM_GPUS))).split(",")
    assert len(all_gpus) >= NUM_GPUS, f"Expected at least {NUM_GPUS} GPUs, got {len(all_gpus)}"
    train_gpus = all_gpus[:NUM_TRAIN_GPUS]
    teacher_gpu = all_gpus[NUM_TRAIN_GPUS]  # 1 GPU is enough for the 0.5B teacher
    return train_gpus, teacher_gpu


def _launch_teacher_server(teacher_gpu: str):
    """Launch an sglang teacher server on the specified GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = teacher_gpu

    log_path = "/tmp/sglang_teacher.log"
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            f"/root/models/{MODEL_NAME}",
            "--host",
            "0.0.0.0",
            "--port",
            str(TEACHER_PORT),
            "--tp",
            "1",
            "--mem-fraction-static",
            "0.6",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    print(f"Starting teacher sglang server on GPU {teacher_gpu} (pid={process.pid}), log: {log_path}")

    # Wait for server to be ready (up to 10 minutes)
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError(f"Teacher server process exited with code {process.returncode}. Check {log_path}")
        try:
            req = urllib.request.urlopen(f"http://{TEACHER_HOST}:{TEACHER_PORT}/health_generate", timeout=2)
            if req.status == 200:
                print(f"Teacher sglang server is ready on GPU {teacher_gpu}")
                return process
        except Exception:
            pass
        time.sleep(5)

    process.kill()
    raise RuntimeError(f"Teacher server failed to start within timeout. Check {log_path}")


def execute():
    train_gpus, teacher_gpu = _get_gpu_split()
    teacher_process = None

    # Restrict CUDA_VISIBLE_DEVICES to training GPUs before Ray starts
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)

    def launch_teacher():
        nonlocal teacher_process
        teacher_process = _launch_teacher_server(teacher_gpu)

    try:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

        rollout_args = (
            "--prompt-data /root/datasets/gsm8k/train.parquet "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--num-rollout 2 "
            "--rollout-batch-size 8 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--global-batch-size 32 "
        )

        eval_args = (
            "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
        )

        rm_args = (
            "--custom-rm-path slime.rollout.on_policy_distillation.reward_func "
            "--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards "
            f"--rm-url http://{TEACHER_HOST}:{TEACHER_PORT}/generate "
        )

        grpo_args = (
            "--advantage-estimator grpo "
            # OPD with sglang teacher (self-distillation for CI test)
            "--use-opd "
            "--opd-type sglang "
            "--opd-kl-coef 1.0 "
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

        sglang_args = (
            "--rollout-num-gpus-per-engine 1 "
            f"--sglang-mem-fraction-static {0.6 if TIGHT_DEVICE_MEMORY else 0.7} "
            "--sglang-enable-metrics "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
            "--colocate "
            "--megatron-to-hf-mode bridge "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{eval_args} "
            f"{sglang_args} "
            f"{ci_args} "
            f"{misc_args} "
            f"{rm_args} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_TRAIN_GPUS,
            megatron_model_type=MODEL_TYPE,
            before_ray_job_submit=launch_teacher,
        )
    finally:
        if teacher_process:
            teacher_process.kill()
            teacher_process.wait()
        U.exec_command("pkill -9 sglang; true")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
