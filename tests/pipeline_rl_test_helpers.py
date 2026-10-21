"""Live-request probes used by the PipelineRL GPU e2e test."""

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
import torch.distributed as dist

_probes = []
_records = {}
_worker_urls_written = False
_initial_weights = None


async def reward(args, sample, **kwargs):
    # The serving address is assigned inside RolloutManager, after training
    # actors receive their args. Publish it from the rollout-side fixture.
    global _worker_urls_written
    if not _worker_urls_written:
        response = requests.get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers", timeout=10)
        response.raise_for_status()
        urls = sorted({worker["url"] for worker in response.json()["workers"]})
        assert urls
        path = Path(os.environ["SLIME_PIPELINE_RL_PROBE_FILE"]).with_suffix(".workers.json")
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(urls))
        tmp.replace(path)
        _worker_urls_written = True
    # Give each GRPO group deterministic nonuniform rewards so this short
    # infrastructure test exercises a real policy gradient even on truncated
    # math answers. The task's mathematical accuracy is not under test.
    return float(sample.index % args.n_samples_per_prompt)


class GenerationProbe:
    def __init__(self, url):
        self.url = url
        self.rid = f"pipeline-rl-probe-{uuid.uuid4().hex}"
        self.events = []
        self.condition = threading.Condition()
        self.error = None
        self.done = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            with requests.post(
                f"{self.url}/generate",
                json={
                    "rid": self.rid,
                    "text": "Count the integers from 1 to 10000, one at a time: 1, 2, 3,",
                    "sampling_params": {"temperature": 0.8, "max_new_tokens": 32700, "ignore_eos": True},
                    "stream": True,
                },
                stream=True,
                timeout=(10, 300),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=4096):
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    meta = data["meta_info"]
                    assert meta["id"] == self.rid
                    with self.condition:
                        self.events.append(
                            {
                                "tokens": meta["completion_tokens"],
                                "version": str(meta["weight_version"]),
                                "finish_reason": meta.get("finish_reason"),
                                "time": time.time(),
                            }
                        )
                        self.condition.notify_all()
        except Exception as error:
            self.error = error
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def wait(self, predicate):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.error or self.done or predicate(self.events), timeout=180)
            if self.error:
                raise self.error
            assert ready and predicate(self.events), f"Probe failed on {self.url}: {self.events[-3:]}"

    def verify(self):
        first = self.events[0]
        self.wait(
            lambda events: any(e["version"] != first["version"] and e["tokens"] > first["tokens"] + 16 for e in events)
        )
        # Every event up to this point belongs to the same uninterrupted HTTP
        # request. An abort/restart or a disk reload's implicit flush must fail.
        assert all(e["finish_reason"] is None for e in self.events), self.events[-3:]
        latest = self.events[-1].copy()
        response = requests.get(f"{self.url}/get_weight_version", timeout=10)
        response.raise_for_status()
        assert str(response.json()["weight_version"]) == latest["version"]
        # End only our synthetic probe after it has proved continuity.
        response = requests.post(f"{self.url}/abort_request", json={"rid": self.rid}, timeout=10)
        response.raise_for_status()
        self.thread.join(timeout=30)
        assert not self.thread.is_alive()
        return {"url": self.url, "rid": self.rid, "before": first, "after": latest}

    def verify_flush(self):
        self.wait(lambda events: bool(events) and events[-1]["finish_reason"] is not None)
        self.thread.join(timeout=30)
        assert not self.thread.is_alive()
        latest = self.events[-1].copy()
        assert latest["finish_reason"]["type"] == "abort", self.events[-3:]
        response = requests.get(f"{self.url}/get_weight_version", timeout=10)
        response.raise_for_status()
        return {
            "url": self.url,
            "rid": self.rid,
            "before": self.events[0],
            "after": latest,
            "serving_version": str(response.json()["weight_version"]),
        }


def start_probes(urls, *, pause=False):
    _probes.clear()
    for url in urls:
        probe = GenerationProbe(url)
        _probes.append(probe)
        probe.thread.start()
    for probe in _probes:
        probe.wait(lambda events: bool(events) and events[-1]["tokens"] >= 16)
        if pause:
            # Keep the live request across slow first-step compilation. The
            # first weight update resumes it without flushing the cache.
            response = requests.post(f"{probe.url}/pause_generation", json={"mode": "in_place"}, timeout=10)
            response.raise_for_status()


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Start a long request on every serving engine before the first update."""
    if dist.get_rank() != 0:
        return
    global _initial_weights
    if rollout_id == 0 and step_id == 0:
        name, param = next((name, param) for name, param in model[0].named_parameters() if "linear_qkv.weight" in name)
        _initial_weights = (name, param.detach().cpu().clone())
        path = Path(os.environ["SLIME_PIPELINE_RL_PROBE_FILE"]).with_suffix(".workers.json")
        urls = json.loads(path.read_text())
        assert urls
        start_probes(urls, pause=True)
    elif rollout_id == 1 and step_id == 0:
        assert _probes
        name, initial = _initial_weights
        current = dict(model[0].named_parameters())[name].detach().cpu()
        assert not torch.equal(initial, current), "The policy weights did not change after training"
        _records["weights_changed"] = True
        with ThreadPoolExecutor(max_workers=len(_probes)) as pool:
            _records["continuous"] = list(pool.map(lambda probe: probe.verify(), _probes))
        start_probes([probe.url for probe in _probes])
    elif rollout_id == 2 and step_id == 0:
        assert _probes
        with ThreadPoolExecutor(max_workers=len(_probes)) as pool:
            _records["next_update"] = list(
                pool.map(
                    lambda probe: probe.verify_flush() if args.flush_cache_interval == 2 else probe.verify(), _probes
                )
            )
        path = Path(os.environ["SLIME_PIPELINE_RL_PROBE_FILE"])
        path.write_text(json.dumps(_records, indent=2))
