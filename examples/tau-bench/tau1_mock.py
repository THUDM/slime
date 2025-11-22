import argparse
import json
import os

from tau_bench.envs import get_env
from tau_bench.types import RunConfig

ALL_DATA_MAPPINGS = {"retail": ["train", "test", "dev"], "airline": ["test"]}


def main():
    parser = argparse.ArgumentParser(description="Tau1 Mock Script")
    parser.add_argument("--local_dir", required=True, help="Path to the local directory")
    args = parser.parse_args()

    local_dir = args.local_dir
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    config = RunConfig(model_provider="mock", user_model_provider="mock", user_strategy="human", model="mock")
    for env, split in ALL_DATA_MAPPINGS.items():
        for s in split:
            config.env = env
            config.task_split = s
            env_instance = get_env(
                env_name=config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
            )
            output_path = os.path.join(local_dir, f"{env}_{s}_tasks.jsonl")
            with open(output_path, "w") as f:
                for i, task in enumerate(env_instance.tasks):
                    row = {"index": i, "metadata": task.model_dump()}
                    f.write(json.dumps(row) + "\n")  # <-- one JSON object per line
            print(f"Saved preprocessed task indices for {env} ({s}) to {output_path}")


if __name__ == "__main__":
    main()
