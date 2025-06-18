def calculate_embedding_flops(seqlen, hidden_size):
    return 2 * seqlen * hidden_size


def calculate_lm_head_flops(seqlen, hidden_size, vocab_size):
    return 2 * seqlen * hidden_size * vocab_size


def calculate_qkv_projection_flops(seqlen, hidden_size, num_attention_heads, num_query_groups):
    head_dim = hidden_size // num_attention_heads
    n_q_heads = num_attention_heads
    n_kv_heads = num_query_groups
    q_flops = 2 * seqlen * hidden_size * n_q_heads * head_dim
    kv_flops = 2 * seqlen * hidden_size * n_kv_heads * head_dim * 2
    return q_flops + kv_flops


def calculate_attention_flops(seqlen, num_attention_heads, head_dim):
    # QK^T
    flops = 2 * num_attention_heads * seqlen * seqlen * head_dim
    # A*V
    flops += 2 * num_attention_heads * seqlen * seqlen * head_dim
    return flops


def calculate_output_flops(seqlen, hidden_size):
    return 2 * seqlen * hidden_size * hidden_size


def calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size):
    return 2 * seqlen * hidden_size * ffn_hidden_size * 3


def calculate_layer_flops(seqlen, hidden_size, num_attention_heads, num_query_groups, ffn_hidden_size):
    head_dim = hidden_size // num_attention_heads
    return (
        calculate_qkv_projection_flops(seqlen, hidden_size, num_attention_heads, num_query_groups)
        + calculate_attention_flops(seqlen, num_attention_heads, head_dim)
        + calculate_output_flops(seqlen, hidden_size)
        + calculate_mlp_flops(seqlen, hidden_size, ffn_hidden_size)
    )


def calculate_fwd_flops(
    seqlens,
    args,
):
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    vocab_size = args.vocab_size

    total_flops = 0

    dense_ffn = args.ffn_hidden_size
    if args.num_experts is None:
        num_dense_layers = args.num_layers
        num_moe_layers = 0
    else:
        shared_expert_ffn = getattr(args, "moe_shared_expert_intermediate_size", None)
        if shared_expert_ffn is None:
            shared_expert_ffn = 0

        moe_ffn = args.moe_ffn_hidden_size * args.moe_router_topk + shared_expert_ffn
        if hasattr(args, "moe_layer_freq"):
            if isinstance(args.moe_layer_freq, list):
                num_dense_layers = sum(1 for freq in args.moe_layer_freq if freq == 0)
                num_moe_layers = sum(1 for freq in args.moe_layer_freq if freq > 0)
            else:
                num_dense_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq != 0)
                num_moe_layers = sum(1 for i in range(args.num_layers) if i % args.moe_layer_freq == 0)
        else:
            num_dense_layers = 0
            num_moe_layers = args.num_layers

    for seqlen in seqlens:
        if num_dense_layers > 0:
            total_flops += (
                calculate_layer_flops(
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    dense_ffn,
                )
                * num_dense_layers
            )

        if num_moe_layers > 0:
            total_flops += (
                calculate_layer_flops(
                    seqlen,
                    hidden_size,
                    num_attention_heads,
                    num_query_groups,
                    moe_ffn,
                )
                * num_moe_layers
            )

        total_flops += calculate_lm_head_flops(seqlen, hidden_size, vocab_size)

    return total_flops
