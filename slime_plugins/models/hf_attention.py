from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig


class HuggingfaceAttention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        model_comm_pgs=None,
    ):
        super().__init__(config=config)
        self.args = args
        self.config = config
        # Note that megatron layer_number starts at 1
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1
        self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # hardcode to fa2 at the moment.
        self.hf_config._attn_implementation = "flash_attention_2"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert packed_seq_params is not None
        cu_seqlens = packed_seq_params.cu_seqlens_q

        if self.args.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=mpu.get_tensor_model_parallel_group()
            )

        if mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            hidden_states_list = [
                torch.empty_like(hidden_states) for _ in range(mpu.get_context_parallel_world_size())
            ]
            dist.nn.all_gather(
                hidden_states_list,
                hidden_states,
                group=mpu.get_context_parallel_group(),
                async_op=False,
            )

            # TODO: preprocess this for each batch to prevent tolist in the training step
            whole_hidden_states_list = []

            local_cu_seqlens = cu_seqlens // cp_size
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2
                whole_hidden_states_list.extend(
                    [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
                        for cp_rank in range(cp_size)
                    ]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1],
                )
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        position_ids = []
        for i in range(len(cu_seqlens) - 1):
            seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
            chunk_size = seqlen // 2
            position_ids.append(torch.arange(seqlen, device=hidden_states.device))
        position_ids = torch.cat(position_ids, dim=0).unsqueeze(0)
        hidden_states = hidden_states.permute(1, 0, 2)  # [bsz, seq_len, hidden_dim]

        output = self.hf_forward(hidden_states, position_ids, packed_seq_params)
        bias = None

        if mpu.get_context_parallel_world_size() > 1:
            output_list = []
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2

        output = output.permute(1, 0, 2)  # [seq_len, bsz, hidden_dim]

        if self.args.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=mpu.get_tensor_model_parallel_group()
            )

        return output, bias

    @abstractmethod
    def hf_forward(self, hidden_states, position_ids, packed_seq_params):
        """Huggingface forward function"""
