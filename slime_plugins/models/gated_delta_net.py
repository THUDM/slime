import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ReplicaId, ShardedTensorFactory
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.jit import jit_fuser
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

try:
    from fla.modules.convolution import causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    causal_conv1d = None
    l2norm = None
    chunk_gated_delta_rule = None
    HAVE_FLA = False


logger = logging.getLogger(__name__)


@dataclass
class GatedDeltaNetSubmodules:
    in_proj: Union[ModuleSpec, type] = IdentityOp
    out_norm: Union[ModuleSpec, type] = IdentityOp
    out_proj: Union[ModuleSpec, type] = IdentityOp


def _mark_tensor_parallel_parameter(param: torch.nn.Parameter, partition_dim: int) -> None:
    setattr(param, "tensor_model_parallel", True)
    setattr(param, "partition_dim", partition_dim)
    setattr(param, "partition_stride", 1)


class GatedDeltaNet(MegatronModule):
    """Gated Delta Net with TP-sharded weights and CP all-to-all activation layout."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedDeltaNetSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        cp_comm_type: Optional[str] = None,
    ):
        if not HAVE_FLA:
            raise ImportError(
                "FLA is not installed. Please install flash-linear-attention to use GatedDeltaNet."
            )

        super().__init__(config)

        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
        self.pg_collection = pg_collection
        self.cp_comm_type = cp_comm_type
        self.cp_size = self.pg_collection.cp.size()
        self.tp_size = self.pg_collection.tp.size()
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        self.config = config
        self.hidden_size = config.hidden_size
        self.act_fn = config.activation_func
        self.activation = self.act_fn.__name__
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.qk_dim_local_tp = self.qk_dim // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        assert self.num_key_heads % self.tp_size == 0
        assert self.num_value_heads % self.tp_size == 0
        assert self.qk_dim_local_tp % self.cp_size == 0
        assert self.v_dim_local_tp % self.cp_size == 0
        assert (self.num_value_heads // self.tp_size) % self.cp_size == 0

        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        if self.config.fp8:
            fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
            assert self.in_proj_dim % fp8_align_size == 0, (
                "For FP8, the GDN input projection output dimension must be aligned."
            )

        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )

        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            bias=conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        )
        _mark_tensor_parallel_parameter(self.conv1d.weight, 0)
        if conv_bias:
            _mark_tensor_parallel_parameter(self.conv1d.bias, 0)

        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        _mark_tensor_parallel_parameter(self.dt_bias, 0)
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        _mark_tensor_parallel_parameter(self.A_log, 0)

        self.gated_delta_rule = (
            torch_chunk_gated_delta_rule if self.config.deterministic_mode else chunk_gated_delta_rule
        )

        self.out_norm = build_module(
            submodules.out_norm,
            config=self.config,
            hidden_size=self.value_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.out_proj = build_module(
            submodules.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.config.perform_initialization:
            with get_cuda_rng_tracker().fork():
                if self.conv_init is not None:
                    nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
                torch.ones(
                    self.num_v_heads_local_tp,
                    out=self.dt_bias.data,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                )
                A = torch.empty(
                    self.num_v_heads_local_tp,
                    dtype=self.config.params_dtype,
                    device=torch.cuda.current_device(),
                ).uniform_(*self.A_init_range)
                self.A_log.data.copy_(torch.log(A))

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            assert inference_context.is_static_batching(), (
                "GDN does not currently support dynamic inference batching."
            )
            assert not self.config.sequence_parallel
            raise NotImplementedError("GDN does not support inference for now.")

        is_packed_thd = (
            packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd"
        )
        if is_packed_thd:
            assert batch == 1, "Packed sequence expects batch dimension to be 1"
            assert not self.config.deterministic_mode, (
                "Packed sequence does not support deterministic GDN mode."
            )
            cu_seqlens_q = self._resolve_cu_seqlens(
                getattr(packed_seq_params, "cu_seqlens_q_padded", None),
                packed_seq_params.cu_seqlens_q,
                seq_len,
                "cu_seqlens_q",
            )
            cu_seqlens_kv = self._resolve_cu_seqlens(
                getattr(packed_seq_params, "cu_seqlens_kv_padded", None),
                packed_seq_params.cu_seqlens_kv,
                seq_len,
                "cu_seqlens_kv",
            )
            assert torch.equal(cu_seqlens_q, cu_seqlens_kv), (
                "Currently only support cu_seqlens_q equals to cu_seqlens_kv."
            )
            assert cu_seqlens_q.shape[0] > 1, "Number of packed sequences must be greater than 0"
        else:
            cu_seqlens_q = None

        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
            self.v_dim_local_tp,
            self.num_value_heads // self.tp_size,
            self.num_value_heads // self.tp_size,
        ]

        if is_packed_thd:
            outputs = []
            for qkvzba_i in _unpack_sequence(qkvzba, cu_seqlens_q // self.cp_size, dim=0):
                outputs.append(
                    tensor_a2a_cp2hp(
                        qkvzba_i,
                        seq_dim=0,
                        head_dim=-1,
                        cp_group=self.pg_collection.cp,
                        split_sections=split_sections,
                    )
                )
            qkvzba = torch.cat(outputs, dim=0)
        else:
            qkvzba = tensor_a2a_cp2hp(
                qkvzba,
                seq_dim=0,
                head_dim=-1,
                cp_group=self.pg_collection.cp,
                split_sections=split_sections,
            )

        qkvzba = qkvzba.transpose(0, 1)

        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim_local_tp * 2 + self.v_dim_local_tp) // self.cp_size,
                self.v_dim_local_tp // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
                self.num_value_heads // self.tp_size // self.cp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        nvtx_range_push(suffix="conv1d")
        local_seq_len = qkv.shape[1]
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight,
            dim=0,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias,
                dim=0,
                cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            )
            if self.conv_bias
            else None
        )
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                input=qkv,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=self.conv1d.stride,
                padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :local_seq_len])
            qkv = qkv.transpose(1, 2)
        else:
            assert self.activation in ["silu", "swish"]
            qkv, _ = causal_conv1d(
                x=qkv,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=self.activation,
                initial_state=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens_q,
            )
        nvtx_range_pop(suffix="conv1d")

        nvtx_range_push(suffix="prepare_qkv_for_gated_delta_rule")
        query, key, value, gate, beta, alpha = self._prepare_qkv_for_gated_delta_rule(
            qkv, gate, beta, alpha, batch, local_seq_len
        )
        nvtx_range_pop(suffix="prepare_qkv_for_gated_delta_rule")

        nvtx_range_push(suffix="g_and_beta")
        A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=self.pg_collection.cp)
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias, dim=0, cp_group=self.pg_collection.cp
        )
        g, beta = self._compute_g_and_beta(A_log_local_cp, dt_bias_local_cp, alpha, beta)
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        core_attn_out, _ = self.gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens_q,
        )
        nvtx_range_pop(suffix="gated_delta_rule")

        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        norm_out = norm_out.reshape(batch, local_seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()

        if is_packed_thd:
            outputs = []
            for norm_out_i in _unpack_sequence(norm_out, cu_seqlens_q, dim=0):
                outputs.append(
                    tensor_a2a_hp2cp(norm_out_i, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)
                )
            norm_out = torch.cat(outputs, dim=0)
        else:
            norm_out = tensor_a2a_hp2cp(
                norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp
            )

        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        x_dtype = x.dtype
        x = x.reshape(-1, x.shape[-1])
        y = self.out_norm(x)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * self.act_fn(gate.float())
        return y.to(x_dtype)

    @jit_fuser
    def _prepare_qkv_for_gated_delta_rule(self, qkv, gate, beta, alpha, batch, seq_len):
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        if self.use_qk_l2norm:
            query_key = l2norm(query_key.contiguous())

        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)
        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        return (
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.contiguous(),
            alpha.contiguous(),
        )

    @jit_fuser
    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        g = -A_log_local_cp.float().exp() * F.softplus(alpha.float() + dt_bias_local_cp.float())
        beta = beta.sigmoid()
        return g, beta

    def _resolve_cu_seqlens(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name):
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
        total_cu = cu_seqlens[-1].item()
        if total_cu != total_seq_len:
            raise ValueError(
                f"GDN: {name}[-1]={total_cu} does not match total_sequence_length={total_seq_len}."
            )
        return cu_seqlens

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)

        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
            sharded_offsets=sharded_offsets,
            tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
            dp_cp_group=metadata["dp_cp_group"],
        )

        tp_group = tp_group if tp_group is not None else self.pg_collection.tp
        for name, module in self.named_children():
            if name == "conv1d":
                module_sd = module.state_dict(prefix="", keep_vars=True)
                tp_sharding_map = {"weight": 0}
                if self.conv_bias:
                    tp_sharding_map["bias"] = 0
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module_sd,
                    f"{prefix}{name}.",
                    tp_sharding_map,
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)

        in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp
        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            [
                self.qk_dim_local_tp,
                self.qk_dim_local_tp,
                self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )

        conv_layer_name_list = ["conv1d.weight"]
        if self.conv_bias:
            conv_layer_name_list.append("conv1d.bias")
        for conv_layer_name in conv_layer_name_list:
            sharded_state_dict[f"{prefix}{conv_layer_name}"] = _split_tensor_factory(
                sharded_state_dict[f"{prefix}{conv_layer_name}"],
                [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
                ["query", "key", "value"],
                0,
            )

        return sharded_state_dict

    def backward_dw(self):
        self.in_proj.backward_dw()
        self.out_proj.backward_dw()


def _unpack_sequence(x, cu_seqlens, dim=1):
    unpacked_x = []
    for i in range(cu_seqlens.shape[0] - 1):
        idx_start = cu_seqlens[i].item()
        idx_end = cu_seqlens[i + 1].item()
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()

    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, got {split_sections=} "
            f"vs {orig_sh_ten_no_data.local_shape[split_dim]=}"
        )
    assert len(split_sections) == len(split_names)

    @torch.no_grad()
    def sh_ten_build_fn(key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]):
        factory_sh_ten = replace(
            orig_sh_ten_no_data,
            key=key,
            data=t,
            dtype=t.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )

        chunk_sh_tens = []
        split_start = 0
        for split_size, split_name in zip(split_sections, split_names):
            split_chunks = factory_sh_ten.narrow(split_dim, split_start, split_size)
            for sh_ten in split_chunks:
                sh_ten.key = f"{sh_ten.key}.{split_name}"
            chunk_sh_tens.extend(split_chunks)
            split_start += split_size

        assert split_start == orig_sh_ten_no_data.local_shape[split_dim]
        assert sum(sh_ten.data.numel() for sh_ten in chunk_sh_tens) == t.numel()
        return chunk_sh_tens

    @torch.no_grad()
    def sh_ten_merge_fn(sub_state_dict):
        return torch.cat(sub_state_dict)

    return ShardedTensorFactory(
        orig_sh_ten.key, orig_sh_ten.data, sh_ten_build_fn, sh_ten_merge_fn, orig_sh_ten.replica_id
    )


def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    if cp_size == 1:
        return param

    if split_sections is not None:
        outputs = [
            get_parameter_local_cp(p, dim, cp_group) for p in torch.split(param, split_sections, dim=dim)
        ]
        return torch.cat(outputs, dim=dim)

    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    return param[slices]


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor

    assert seq_dim == 0
    assert head_dim == -1 or head_dim == 2
    assert tensor.dim() == 3

    if split_sections is not None:
        outputs = [
            tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            for x in torch.split(tensor, split_sections, dim=head_dim)
        ]
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    cp_size = cp_group.size()
    if cp_size == 1:
        return tensor

    assert seq_dim == 0
    assert head_dim == -1 or head_dim == 2
    assert tensor.dim() == 3

    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    if split_sections is not None:
        outputs = [
            tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            for x in torch.split(tensor, split_sections, dim=head_dim)
        ]
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    assert cu_seqlens is None, "cu_seqlens is not supported for deterministic GDN mode."

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    sequence_length = key.shape[2]
    k_head_dim = key.shape[-1]
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=query.device)
    attn = attn.to(torch.bfloat16)
    value = attn @ v_beta
    k_cumsum = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = None
    core_attn_out = torch.zeros_like(value)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        if initial_state is not None:
            core_attn_out[:, :, i] += (q_i * g[:, :, i, :, None].exp()) @ initial_state
        core_attn_out[:, :, i] += (q_i @ k_i.transpose(-1, -2)).tril() @ v_i
        initial_state = (
            initial_state * g[:, :, i, -1, None, None].exp()
            if initial_state is not None
            else torch.zeros_like(query[:, :, 0].transpose(-1, -2) @ value[:, :, 0])
        )
        initial_state = initial_state + (
            k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_i
        initial_state = initial_state - k_cumsum[:, :, i].transpose(-1, -2) @ v_i
    if output_final_state:
        last_recurrent_state = initial_state

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, v_head_dim
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
