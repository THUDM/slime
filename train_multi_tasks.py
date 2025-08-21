
import os
import torch
import ray
import asyncio
import argparse


from slime.scheduler import Scheduler, SchedulerParams
from slime.utils.arguments import parse_multi_task_args

async def main(tasks_args):
    if not ray.is_initialized():
        print("--"*50)
        # ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "RAY_DEBUG_POST_MORTEM": "1"}})
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    scheduler_args = SchedulerParams(
        tasks_num=len(tasks_args),
        tasks_args_template=tasks_args,
        train_semaphore=1,
        rollout_semaphore=1,
    )
    scheduler = Scheduler(scheduler_args)

    await scheduler.init_task()
    await scheduler.run()


if __name__ == "__main__":
    task_configs = os.environ.get("TASK_CONFIGS")
    task_config_files = task_configs.split()
    tasks_args = parse_multi_task_args(task_config_files)
    for task_args in tasks_args:
        print("*"*100)
        print(f"{task_args.use_dynamic_batch_size}")
        print("*"*100)
    asyncio.run(main(tasks_args))


    print("finish")
