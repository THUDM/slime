"""Traversal for the processor-output dicts that ride along a multimodal sample.

``Sample.multimodal_train_inputs`` is whatever the HF processor returned for one
sample minus the token stream -- ``pixel_values``, ``image_grid_thw``,
``num_patches``, and so on -- and its values are not uniformly tensors. A
fixed-resolution processor (Qwen3.5-VL) returns one stacked tensor per field. A
dynamic-resolution one (Nemotron 3.5 Super VL) resizes every image to its own
aspect-preserving patch grid, so it returns ``pixel_values`` as a *list* of
``(3, H, W)`` tensors whenever two images in the sample differ in shape, next to
plain Python lists such as ``num_patches`` / ``num_tokens``.

The list form is not a defect to normalize away: the model's own vision
projector iterates it, because per-image feature counts differ and cannot be
stacked. So every stage that walks one of these dicts has to look *inside* the
containers rather than test the top-level value with ``isinstance(value,
torch.Tensor)`` and pass everything else through untouched -- that test silently
leaves a list of per-image tensors on the host, and the projector then meets CPU
inputs with CUDA weights.

The micro-batch join in ``slime.backends.megatron_utils.data`` is the same
contract seen from the other side: it dispatches on type instead of assuming
``torch.cat`` applies.
"""

__all__ = ["map_multimodal_fields"]


def map_multimodal_fields(mm_inputs, fn):
    """Rebuild one processor-output dict with ``fn`` applied to every leaf.

    Lists and tuples are traversed and their container type preserved, so a
    ``pixel_values`` list of per-image tensors is converted element by element
    and stays a list, which is the form the vision projector takes.

    ``fn`` sees only leaves and decides for itself what to touch. Both call
    sites keep a tensor-only guard there, which is what leaves ``num_patches =
    [3, 5]`` a list of Python ints instead of promoting it to a tensor.

    ``None`` passes through, because a micro-batch may mix text-only samples
    (whose ``multimodal_train_inputs`` is ``None``) with multimodal ones.
    """
    if mm_inputs is None:
        return None
    return {key: _map_field(value, fn) for key, value in mm_inputs.items()}


def _map_field(value, fn):
    if isinstance(value, list):
        return [_map_field(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_field(item, fn) for item in value)
    return fn(value)
