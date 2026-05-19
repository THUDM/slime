import sys
import types

import numpy as np
import torch


class TensorDict(dict):
    def __init__(self, source=None, batch_size=None, device=None):
        super().__init__(source or {})
        self.batch_size = torch.Size(batch_size or [])
        self.device = device

    def __len__(self):
        return self.batch_size[0] if len(self.batch_size) > 0 else dict.__len__(self)

    def clone(self):
        return TensorDict({key: val.clone() for key, val in self.items()}, self.batch_size, self.device)

    def select(self, *keys):
        return TensorDict({key: self[key] for key in keys}, self.batch_size, self.device)


tensordict_module = types.ModuleType("tensordict")
tensordict_td_module = types.ModuleType("tensordict._td")
tensordict_module.TensorDict = TensorDict
tensordict_td_module.TensorDict = TensorDict
sys.modules.setdefault("tensordict", tensordict_module)
sys.modules.setdefault("tensordict._td", tensordict_td_module)

from slime.utils.remote_batch import MooncakeRemoteBatch, create_mooncake_store, normalize_store_init_kwargs
from slime.utils.rollout_dataproto import DataProto, dataproto_to_rollout_data, split_rollout_data_by_dp_dataproto


class FakeRemoteBatch:
    def __init__(self, tensors, indices=None):
        self.tensors = tensors
        self.indices = list(range(len(next(iter(tensors.values()))))) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    @property
    def batch_size(self):
        return torch.Size([len(self.indices)])

    def keys(self):
        return list(self.tensors.keys())

    def materialize(self, fields=None):
        selected = self.keys() if fields is None else fields
        return TensorDict({key: self.tensors[key][self.indices] for key in selected}, batch_size=(len(self),))


def test_dataproto_remote_materialize():
    remote = FakeRemoteBatch({"tokens": torch.arange(12).reshape(4, 3), "loss_masks": torch.ones(4, 3, dtype=torch.int32)})
    proto = DataProto.from_remote(remote, non_tensors={"response_lengths": np.asarray([1, 2, 3, 4])})

    proto.materialize_remote_batch()

    assert proto.remote_batch is None
    assert proto.batch["tokens"].tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    assert proto.non_tensor_batch["response_lengths"].tolist() == [1, 2, 3, 4]


def test_dataproto_to_rollout_data_preserves_remote_tensor_rows():
    remote = FakeRemoteBatch(
        {
            "tokens": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "loss_masks": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.int),
        }
    )
    proto = DataProto.from_remote(
        remote,
        non_tensors={"partition": np.asarray([0, 1])},
        meta_info={
            "total_lengths": [2, 3],
            "tokens_lengths": [2, 3],
            "loss_masks_lengths": [2, 3],
        },
    )

    rollout_data = dataproto_to_rollout_data(proto)

    assert all(isinstance(row, torch.Tensor) for row in rollout_data["tokens"])
    assert all(isinstance(row, torch.Tensor) for row in rollout_data["loss_masks"])
    assert [row.tolist() for row in rollout_data["tokens"]] == [[1, 2], [3, 4, 5]]
    assert [row.tolist() for row in rollout_data["loss_masks"]] == [[1, 1], [1, 1, 1]]
    assert "_remote_tensor_owners" not in rollout_data
    assert rollout_data["partition"] == [0, 1]
    assert rollout_data["total_lengths"] == [2, 3]


def test_dataproto_to_rollout_data_legacy_tensor_list_fallback():
    remote = FakeRemoteBatch({"tokens": torch.tensor([[1, 2, 0], [3, 4, 5]])})
    proto = DataProto.from_remote(
        remote,
        non_tensors={"partition": np.asarray([0, 1])},
        meta_info={"total_lengths": [2, 3], "tokens_lengths": [2, 3]},
    )

    rollout_data = dataproto_to_rollout_data(proto, preserve_remote_tensors=False)

    assert rollout_data["tokens"] == [[1, 2], [3, 4, 5]]
    assert "_remote_tensor_owners" not in rollout_data


def test_dataproto_to_rollout_data_keeps_non_remote_tensors_legacy():
    proto = DataProto.from_dict(tensors={"other": torch.tensor([[1, 2], [3, 4]])})

    rollout_data = dataproto_to_rollout_data(proto, preserve_remote_tensors=True)

    assert rollout_data["other"] == [[1, 2], [3, 4]]


def test_rollout_transfer_rejects_partition_mismatch():
    args = types.SimpleNamespace(mooncake_dataproto_store_init_kwargs={"setup_method": "setup_dummy"})
    try:
        split_rollout_data_by_dp_dataproto(args, {}, 2, [[]])
    except ValueError as exc:
        assert "expected 2 partitions" in str(exc)
    else:
        raise AssertionError("partition mismatch should be rejected")


def test_normalizes_empty_mooncake_setup_kwargs_to_setup():
    assert normalize_store_init_kwargs({}) == {"setup_method": "setup"}


def test_rejects_unsafe_mooncake_setup_method():
    try:
        normalize_store_init_kwargs({"setup_method": "remove"})
    except ValueError as exc:
        assert "unsupported Mooncake store setup_method" in str(exc)
    else:
        raise AssertionError("unsafe setup_method should be rejected")


def test_create_store_normalizes_none_for_default_call(monkeypatch):
    class Store:
        def setup(self):
            return 0

    mooncake_module = types.ModuleType("mooncake")
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = Store
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)

    assert isinstance(create_mooncake_store(), Store)


def test_rejects_invalid_remote_field_name():
    class Store:
        def get_hostname(self):
            return "localhost"

    try:
        MooncakeRemoteBatch.from_tensors({"../tokens": torch.ones(1, 1, dtype=torch.int64)}, Store(), "prefix")
    except ValueError as exc:
        assert "invalid Mooncake tensor field name" in str(exc)
    else:
        raise AssertionError("invalid tensor field name should be rejected")
