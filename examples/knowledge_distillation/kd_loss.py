import os

import torch
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

KD_TEMPERATURE = float(os.environ.get("KD_TEMPERATURE", "1.0"))
KD_TOP_K = int(os.environ.get("KD_TOP_K", "8"))

_topk_data_store = {}


def store_topk_data(samples):
    """Store top-k data indexed by token prefix for loss function retrieval."""
    global _topk_data_store
    for group in samples:
        for s in group:
            if s.train_metadata:
                _topk_data_store[tuple(s.tokens[:20])] = s.train_metadata


def _get_topk_data(tokens):
    key = tuple(tokens[:20].tolist() if hasattr(tokens, "tolist") else tokens[:20])
    return _topk_data_store.get(key)


def sampled_kl_loss(args, batch, logits, sum_of_sample_mean):
    """Forward KL on teacher-sampled tokens (KD_TOP_K=0)."""
    _, log_probs_result = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=True,
        max_seq_lens=batch.get("max_seq_lens"),
    )
    student_lps = log_probs_result["log_probs"]
    entropy = log_probs_result.get("entropy", [])

    kl_terms = []
    for s_lp, t_lp in zip(student_lps, batch["teacher_log_probs"], strict=False):
        kl_terms.append(t_lp.to(s_lp) - s_lp)

    loss = sum_of_sample_mean(torch.cat(kl_terms))
    log = {"kd/loss": loss.detach()}
    if entropy:
        log["kd/entropy"] = sum_of_sample_mean(torch.cat(entropy)).detach()
    return loss, log


def _extract_response_log_probs(logits, unconcat_tokens, total_lengths, response_lengths):
    results = []
    packed = logits.shape[0] == 1 and len(unconcat_tokens) > 1
    offset = 0
    for i in range(len(unconcat_tokens)):
        total_len, resp_len = int(total_lengths[i]), int(response_lengths[i])
        prompt_len = total_len - resp_len
        if packed:
            row = logits[0, offset + prompt_len - 1 : offset + total_len - 1]
            offset += total_len
        else:
            row = logits[i, prompt_len - 1 : total_len - 1]
        results.append(torch.log_softmax(row.float(), dim=-1))
    return results


def topk_kl_loss(args, batch, logits, sum_of_sample_mean):
    """Forward KL on teacher's top-K tokens with temperature scaling."""
    student_full_lps = _extract_response_log_probs(
        logits,
        batch["unconcat_tokens"],
        batch["total_lengths"],
        batch["response_lengths"],
    )

    topk_data_list = [_get_topk_data(tokens) for tokens in batch["unconcat_tokens"]]
    valid_data = [
        (s_lp, data) for s_lp, data in zip(student_full_lps, topk_data_list, strict=False) if data is not None
    ]

    if not valid_data:
        return sampled_kl_loss(args, batch, logits, sum_of_sample_mean)

    tau = KD_TEMPERATURE
    kl_terms = []
    for s_lp, data in valid_data:
        t_ids = torch.tensor(data["teacher_top_k_ids"], device=s_lp.device, dtype=torch.long)
        t_lps = torch.tensor(data["teacher_top_k_logprobs"], device=s_lp.device, dtype=s_lp.dtype)

        s_topk = s_lp.gather(1, t_ids)
        t_renorm = torch.log_softmax(t_lps / tau, dim=-1)
        s_renorm = torch.log_softmax(s_topk / tau, dim=-1)
        kl_terms.append((tau**2) * (t_renorm.exp() * (t_renorm - s_renorm)).sum(dim=-1))

    loss = sum_of_sample_mean(torch.cat(kl_terms))
    return loss, {"kd/loss": loss.detach()}


def kd_loss_function(args, batch, logits, sum_of_sample_mean):
    """KD loss: top-K KL if KD_TOP_K > 0, else sampled KL."""
    if KD_TOP_K > 0:
        return topk_kl_loss(args, batch, logits, sum_of_sample_mean)
    return sampled_kl_loss(args, batch, logits, sum_of_sample_mean)
