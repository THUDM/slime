import pytest
import torch

pytest.importorskip("megatron")

from slime.backends.megatron_utils.update_weight import update_weight_from_distributed


NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class _Engine:
    def __init__(self):
        self.update_weights_from_distributed = _RemoteMethod()


class _Group:
    def rank(self):
        return 0


def test_p2p_weight_update_matches_engine_receive_protocol(monkeypatch):
    monkeypatch.setenv("UPDATE_MODE", "p2p-broadcast")
    monkeypatch.setattr(update_weight_from_distributed.dist, "get_global_rank", lambda group, rank: rank)

    sends = []
    monkeypatch.setattr(
        update_weight_from_distributed.dist,
        "send",
        lambda tensor, dst, group, tag: sends.append((tensor, dst, group, tag)),
    )
    monkeypatch.setattr(
        update_weight_from_distributed.dist,
        "broadcast",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected collective broadcast")),
    )

    group = _Group()
    engine = _Engine()
    tensors = [("a", torch.ones(2)), ("b", torch.ones(3))]

    refs = update_weight_from_distributed.update_weights_from_distributed("slime-pp_0", group, 1, [engine], tensors)

    assert refs == engine.update_weights_from_distributed.calls
    assert [(dst, send_group, tag) for _, dst, send_group, tag in sends] == [
        (1, group, 0),
        (1, group, 1),
    ]


def test_default_weight_update_keeps_async_broadcast(monkeypatch):
    monkeypatch.delenv("UPDATE_MODE", raising=False)

    waited = []

    class _Handle:
        def wait(self):
            waited.append(True)

    broadcasts = []
    monkeypatch.setattr(
        update_weight_from_distributed.dist,
        "broadcast",
        lambda tensor, src, group, async_op: broadcasts.append((tensor, src, group, async_op)) or _Handle(),
    )

    group = _Group()
    tensors = [("a", torch.ones(2)), ("b", torch.ones(3))]
    update_weight_from_distributed.update_weights_from_distributed("slime-pp_0", group, 1, [_Engine()], tensors)

    assert [(src, broadcast_group, async_op) for _, src, broadcast_group, async_op in broadcasts] == [
        (0, group, True),
        (0, group, True),
    ]
    assert waited == [True, True]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
