"""The traversal contract every stage that walks a processor-output dict relies on.

Deliberately free of torch: ``map_multimodal_fields`` only decides where the
leaves are, and the two call sites supply the conversion. Marking the leaves
with a sentinel rather than a tensor is what makes "which values did the walk
actually reach" the thing being asserted.
"""

from slime.utils.multimodal import map_multimodal_fields

NUM_GPUS = 0


class Leaf:
    """Stand-in for a tensor: convertible, and identifiable after conversion."""

    def __init__(self, name: str, moved: bool = False) -> None:
        self.name = name
        self.moved = moved

    def move(self) -> "Leaf":
        return Leaf(self.name, moved=True)

    def __repr__(self) -> str:  # pragma: no cover - failure messages only
        return f"Leaf({self.name!r}, moved={self.moved})"


def move_leaves(value):
    """What both call sites pass: a converter that guards on the leaf type."""
    return value.move() if isinstance(value, Leaf) else value


# ------------------------------------------------- the dynamic-resolution case --
def test_a_list_of_per_image_tensors_is_converted_elementwise():
    """The regression this module exists for.

    A dynamic-resolution processor returns ``pixel_values`` as a list of
    per-image tensors. Guarding on the top-level value with ``isinstance(value,
    Tensor)`` skips that list whole, which used to leave it on the host while
    the projector ran with CUDA weights.
    """
    result = map_multimodal_fields({"pixel_values": [Leaf("a"), Leaf("b")]}, move_leaves)

    assert isinstance(result["pixel_values"], list)
    assert [leaf.name for leaf in result["pixel_values"]] == ["a", "b"]
    assert all(leaf.moved for leaf in result["pixel_values"])


def test_a_stacked_tensor_is_still_converted():
    """The fixed-resolution case, unchanged: one tensor per field."""
    result = map_multimodal_fields({"pixel_values": Leaf("stacked")}, move_leaves)

    assert result["pixel_values"].moved


# --------------------------------------------------- what must stay untouched --
def test_python_int_lists_are_not_promoted():
    """``num_patches`` / ``num_tokens`` arrive as plain lists and must stay so.

    The guard lives in the converter, not the walk, precisely so the walk can
    descend into this list without the ints coming back as tensors.
    """
    result = map_multimodal_fields({"num_patches": [3, 5], "num_tokens": [256, 1024]}, move_leaves)

    assert result["num_patches"] == [3, 5]
    assert result["num_tokens"] == [256, 1024]


def test_container_types_and_nesting_are_preserved():
    result = map_multimodal_fields({"grids": ([Leaf("a")], (Leaf("b"),))}, move_leaves)

    outer = result["grids"]
    assert isinstance(outer, tuple)
    assert isinstance(outer[0], list) and outer[0][0].moved
    assert isinstance(outer[1], tuple) and outer[1][0].moved


def test_text_only_samples_pass_through():
    """A micro-batch mixes text-only samples, whose dict is ``None``."""
    assert map_multimodal_fields(None, move_leaves) is None


def test_the_input_dict_is_not_mutated():
    """Both call sites rebind the field, so the walk must not edit in place."""
    original = {"pixel_values": [Leaf("a")]}

    map_multimodal_fields(original, move_leaves)

    assert not original["pixel_values"][0].moved
