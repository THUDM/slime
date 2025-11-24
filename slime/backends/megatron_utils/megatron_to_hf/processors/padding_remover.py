import torch


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.
    """
    if name == "module.module.embedding.word_embeddings.weight" or name == "module.module.output_layer.weight":
        return param[:vocab_size]
    return param
