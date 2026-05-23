import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from slime.utils import trackio_utils
from slime.utils.types import Sample


def setup_function():
    trackio_utils._INITIALIZED = False


def _args(**overrides):
    values = {
        "use_trackio": True,
        "trackio_project": "slime-trackio-test",
        "trackio_run_name": "smoke-run",
        "trackio_max_traces_per_rollout": 4,
        "wandb_project": None,
        "wandb_group": None,
        "rank": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_init_trackio_uses_project_run_and_config():
    fake_trackio = MagicMock()

    with patch.dict(sys.modules, {"trackio": fake_trackio}):
        trackio_utils.init_trackio(_args())

    fake_trackio.init.assert_called_once()
    init_kwargs = fake_trackio.init.call_args.kwargs
    assert init_kwargs["project"] == "slime-trackio-test"
    assert init_kwargs["name"] == "smoke-run"
    assert init_kwargs["config"]["trackio_project"] == "slime-trackio-test"


def test_log_rollout_traces_writes_trackio_trace_records():
    fake_trackio = MagicMock()
    fake_trackio.Trace.side_effect = lambda messages, metadata: {
        "_type": "trackio.trace",
        "messages": messages,
        "metadata": metadata,
    }
    sample = Sample(
        index=7,
        group_index=3,
        prompt=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        response="4",
        reward=1.0,
        response_length=1,
        metadata={"source": "unit-test"},
    )
    sample.trace = {"trace_id": "trace-1", "events": [{"name": "sglang_generate"}]}

    with patch.dict(sys.modules, {"trackio": fake_trackio}):
        trackio_utils.log_rollout_traces(_args(), rollout_id=5, samples=[sample], split="rollout", step=9)

    fake_trackio.init.assert_called_once()
    fake_trackio.Trace.assert_called_once()
    trace_kwargs = fake_trackio.Trace.call_args.kwargs
    assert trace_kwargs["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert trace_kwargs["metadata"]["split"] == "rollout"
    assert trace_kwargs["metadata"]["rollout_id"] == 5
    assert trace_kwargs["metadata"]["step"] == 9
    assert trace_kwargs["metadata"]["sample_index"] == 7
    assert trace_kwargs["metadata"]["group_index"] == 3
    assert trace_kwargs["metadata"]["reward"] == 1.0
    assert trace_kwargs["metadata"]["metadata"] == {"source": "unit-test"}
    assert trace_kwargs["metadata"]["trace"] == {"trace_id": "trace-1", "events": [{"name": "sglang_generate"}]}
    fake_trackio.log.assert_called_once()
    log_payload = fake_trackio.log.call_args.args[0]
    assert log_payload["rollout/trajectories"] == [
        {
            "_type": "trackio.trace",
            "messages": trace_kwargs["messages"],
            "metadata": trace_kwargs["metadata"],
        }
    ]
    assert fake_trackio.log.call_args.kwargs == {"step": 9}


def test_log_rollout_traces_respects_max_traces_per_rollout():
    fake_trackio = MagicMock()
    fake_trackio.Trace.side_effect = lambda messages, metadata: {"messages": messages, "metadata": metadata}
    samples = [Sample(index=index, prompt=f"prompt {index}", response="ok") for index in range(3)]

    with patch.dict(sys.modules, {"trackio": fake_trackio}):
        trackio_utils.log_rollout_traces(
            _args(trackio_max_traces_per_rollout=2),
            rollout_id=1,
            samples=samples,
            split="rollout",
            step=0,
        )

    assert fake_trackio.Trace.call_count == 2
