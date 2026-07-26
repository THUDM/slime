"""CPU unit tests for the ``path@[start:end]`` dataset-slicing syntax.

``_parse_generalized_path``'s regex explicitly accepts a sign on both bounds
(``-?\\d*``), and the feature shipped with full negative-index support
(``df.iloc[row_slice]``). The streaming rewrite swapped that for
``itertools.islice``, which raises ``ValueError`` on any negative index — so
``@[-100:]`` ("the last 100 rows") went from working to crashing while the
parser still advertised it.

Pinned here: non-negative slices keep streaming through ``islice``; slices
with a negative bound resolve against the real row count; parsing itself.
"""

from __future__ import annotations

import json

import pytest

from slime.utils.data import _parse_generalized_path, read_file


NUM_GPUS = 0

ROWS = [{"id": i} for i in range(10)]


@pytest.fixture
def jsonl_path(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
    return str(path)


def _ids(generalized_path):
    return [row["id"] for row in read_file(generalized_path)]


@pytest.mark.unit
def test_parse_generalized_path():
    assert _parse_generalized_path("/a/b.jsonl") == ("/a/b.jsonl", None)
    assert _parse_generalized_path("/a/b.jsonl@[3:7]") == ("/a/b.jsonl", slice(3, 7))
    assert _parse_generalized_path("/a/b.jsonl@[-100:]") == ("/a/b.jsonl", slice(-100, None))
    assert _parse_generalized_path("/a/b.jsonl@[:-2]") == ("/a/b.jsonl", slice(None, -2))


@pytest.mark.unit
def test_no_slice_reads_everything(jsonl_path):
    assert _ids(jsonl_path) == list(range(10))


@pytest.mark.unit
@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("@[0:3]", [0, 1, 2]),
        ("@[3:]", [3, 4, 5, 6, 7, 8, 9]),
        ("@[:4]", [0, 1, 2, 3]),
    ],
)
def test_non_negative_slices(jsonl_path, suffix, expected):
    assert _ids(jsonl_path + suffix) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("@[-3:]", [7, 8, 9]),
        ("@[:-2]", [0, 1, 2, 3, 4, 5, 6, 7]),
        ("@[1:-1]", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("@[-5:-2]", [5, 6, 7]),
    ],
)
def test_negative_slices(jsonl_path, suffix, expected):
    assert _ids(jsonl_path + suffix) == expected


@pytest.mark.unit
def test_negative_slice_larger_than_file(jsonl_path):
    # "@[-100:]" on a 10-row file is simply the whole file, like list slicing.
    assert _ids(jsonl_path + "@[-100:]") == list(range(10))
