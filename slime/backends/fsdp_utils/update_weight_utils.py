import ray
import torch
import torch.distributed as dist
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

try:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

    use_flattened_tensor_bucket = True
except:
    use_flattened_tensor_bucket = False


class UpdateWeightFromTensor:
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self):
        monkey_patch_torch_reductions()
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            named_tensors = [(name, param) for name, param in self.model.state_dict().items()]

        if use_flattened_tensor_bucket:
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()

            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            serialized_tensors = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
        else:
            serialized_tensors = MultiprocessingSerializer.serialize(named_tensors, output_str=True)

        serialized_named_tensors = (
            [None] * dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            serialized_tensors,
            object_gather_list=serialized_named_tensors,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            kwargs = {
                "serialized_named_tensors": serialized_named_tensors,
            }
            if use_flattened_tensor_bucket:
                kwargs["load_format"] = "flattened_bucket"

            ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
            ray.get(ref)
