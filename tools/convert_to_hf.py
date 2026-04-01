import torch
import torch.distributed as dist
from megatron.core import mpu
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import slime.backends.megatron_utils as megatron_utils
from slime.backends.megatron_utils import update_weight_utils
from slime.utils.arguments import parse_args


def add_checkpoint_args(parser):
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the converted HF model.",
    )
    parser.add_argument(
        "--check-same",
        action="store_true",
        default=False,
        help="Check if the converted model is the same as the original model.",
    )
    return parser


def main(args):
    megatron_utils.init(args)

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    is_save_rank = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    # Setup the model and optimizer
    args.no_load_optim = True
    args.no_load_rng = True
    model, _, _, _ = megatron_utils.initialize_model_and_optimizer(args)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    model_name = type(hf_config).__name__.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    vocab_size = tokenizer.vocab_size if args.vocab_size is None else args.vocab_size

    param_infos = update_weight_utils.get_param_infos(args, model)

    state_dict = {}
    rank = dist.get_rank()
    for info in param_infos:
        if dist.get_rank() == info.src_rank:
            for name_, param_ in update_weight_utils.named_parameters(args, model):
                if name_ == info.name:
                    param = param_
                    break
        else:
            param = torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device())

        if pp_size > 1:
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                torch.distributed.broadcast(param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group())

        # broadcast params across ep ranks
        if ep_size > 1:
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                torch.distributed.broadcast(param, src=src_rank, group=mpu.get_expert_model_parallel_group())

        for key, value in info.attrs.items():
            setattr(param, key, value)

        param = update_weight_utils.all_gather_param(info.name, param)
        param = update_weight_utils.remove_padding(info.name, param, vocab_size)
        # use torch.distributed
        if is_save_rank:
            converted_named_tensors = update_weight_utils.convert_to_hf(args, model_name, info.name, param)
            for name, param in converted_named_tensors:
                state_dict[name] = param.cpu()
        del param

    if is_save_rank:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_checkpoint, torch_dtype="auto", device_map="cpu", trust_remote_code=True
        )

        if args.check_same:
            for name, param in hf_model.named_parameters():
                if name in state_dict:
                    assert (
                        param.shape == state_dict[name].shape
                    ), f"Shape mismatch for {name}: {param.shape} vs {state_dict[name].shape}"
                    assert torch.all(param == state_dict[name]), f"Value mismatch for {name}"
                else:
                    print(f"Warning: {name} not found in state_dict")

        if args.output_dir:
            tokenizer.save_pretrained(args.output_dir)
            print(hf_model.load_state_dict(state_dict, strict=False))
            hf_model.save_pretrained(args.output_dir)

    dist.barrier()


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_checkpoint_args)
    main(args)
