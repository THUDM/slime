import copy

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Any, Dict

from slime.ray.placement_group import create_placement_groups


@dataclass
class SchedulerParams:
    """The Params about Scheduler"""    
    tasks_num: int = 0
    tasks_args_dir: str = "xxx"
    tasks_args: list = None
    tasks_args_template: Any = None

    pgs: dict = None
    train_semaphore: int = 1
    rollout_semaphore:int  = 1
    update_weight_semaphore:int = 1
    save_semaphore: int = 1

    # TODO:should read multi tasks args from different yaml file
    def __post_init__(self):
        task_args=self.tasks_args_template[0]
        self.pgs = create_placement_groups(task_args)

        for task_args in self.tasks_args_template:
            task_args.pgs = self.pgs
            task_args.tasks_num = self.tasks_num

        # TODO 读取task_args_dir下的list，然后分别整理成参数列表返回
        self.tasks_args= self.tasks_args_template

