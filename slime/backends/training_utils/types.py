from dataclasses import dataclass
import torch.distributed as dist

@dataclass
class ParallelState:
    dp_rank: int
    dp_src_rank: int
    dp_size: int
    cp_rank: int
    cp_size: int
    tp_size: int
    tp_rank: int
    vpp_size: int | None
    microbatch_group_size_per_vp_stage: int | None
    dp_group: dist.ProcessGroup | None
    dp_group_gloo: dist.ProcessGroup | None
    cp_group: dist.ProcessGroup | None
    tp_group: dist.ProcessGroup | None
    dp_mesh: dist.DeviceMesh | None
    cp_mesh: dist.DeviceMesh | None
    is_pp_last_stage: bool

    def __init__(self):
        self.vpp_size = 1
        self.microbatch_group_size_per_vp_stage = None
        self.is_pp_last_stage = True