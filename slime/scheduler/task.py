import ray
import asyncio
from ray.util.placement_group import placement_group
from slime.ray.placement_group import create_actor_group,create_rollout_group

from tracer import vinit, TracePoint, MemTracePoint

class Task:
    _task_id = 0

    def __init__(self, args):
        self.args = args
        self.args.task_id = Task._task_id
        self.task_id = Task._task_id
        Task._task_id = Task._task_id + 1
        self.args.save= f"{self.args.save}/{self.task_id}"
        assert Task._task_id <= self.args.tasks_num
        vinit()
        tp = TracePoint(f"task-{self.task_id}: __init__ task", "1")
        tp.begin()
        self.pgs = args.pgs
        self.tasks_num = args.tasks_num

        amp = TracePoint(f"task-{self.task_id}: create actor group", "1")
        amp.begin()
        self.actor_model = create_actor_group(
                args=self.args,
                pg=self.pgs["actor"]
            )
        amp.end()

        rgp = TracePoint(f"task-{self.task_id}: create rollout group", "1")
        rgp.begin()
        self.rollout_generator = create_rollout_group(
                args=self.args,
                pg=self.pgs["rollout"]
            )
        rgp.end()
        
        tp.end()

    async def init(self):
        # breakpoint()
        tp = TracePoint(f"task-{self.task_id}: init task", "1")
        tp.begin()
        self.args.model_path = None
        self.num_rollout_per_epoch = None
        if self.args.num_rollout is None:
            self.num_rollout_per_epoch = await self.rollout_generator.data_buffer.get_num_rollout_per_epoch.remote()
            self.args.num_rollout = self.num_rollout_per_epoch * self.args.num_epoch
        assert self.args.num_rollout > 0

        # sync the initialization (model initalization, load checkpoint, etc.)
        start_rollout_ids = await asyncio.gather(*(self.actor_model.async_init(
            self.args,role="actor",with_ref=self.args.kl_coef != 0 or self.args.use_kl_loss
        )))
        assert len(set(start_rollout_ids)) == 1
        if self.args.start_rollout_id is None:
            self.args.start_rollout_id = start_rollout_ids[0]

        if self.args.rollout_global_dataset:
            await self.rollout_generator.data_buffer.load.remote(self.args.start_rollout_id - 1)

        # breakpoint()
        await self.actor_model.async_init_weight_update_connections(self.rollout_generator)
        self.start_rollout_ids=start_rollout_ids
        tp.end()