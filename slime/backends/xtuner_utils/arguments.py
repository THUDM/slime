import argparse


def parse_args(add_custom_arguments=None):
    parser = argparse.ArgumentParser(description="SLIME")

    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--rollout-global-batch-size", type=int, default=128)
    parser.add_argument("--train-optimizer-steps", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=512)
    parser.add_argument("--prompt-repeat-k", type=int, default=8)
    parser.add_argument("--pack-max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--optimizer-disable-foreach", action="store_true")  # save memory usage during opt.step()
    parser.add_argument("--policy-loss-type", type=str, default="vanilla")
    parser.add_argument("--enable-evaluate", action="store_true")
    parser.add_argument("--evaluate-step", type=int, default=1)
    parser.add_argument("--evaluate-ratio", type=float, default=1)

    if add_custom_arguments:
        add_custom_arguments(parser)
    args = parser.parse_args()
    return args
