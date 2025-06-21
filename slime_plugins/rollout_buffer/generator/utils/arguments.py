import argparse


def add_arguments(add_task_arguments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="math")
    parser.add_argument("--input_file", type=str, default="None")
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--num_repeat_per_sample", type=int, default=4)
    parser.add_argument("--num_process", type=int, default=4)
    parser.add_argument("--remote_engine_url", type=str, default="http://0.0.0.0:8000/v1")
    parser.add_argument("--remote_buffer_url", type=str, default="http://localhost:8888")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_repeats", type=int, default=20)

    if add_task_arguments is not None:
        parser = add_task_arguments(parser)

    args = parser.parse_args()
    return args
