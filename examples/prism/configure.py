"""Prepare concrete example configs from local assets and frozen services.

This reads /model_info and /info once; it never installs weights or launches
servers. Run from the repository root with the training Python environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml

from slime.backends.sglang_utils.omni_engine import validate_admin_result
from slime_plugins.omni_rollout import PrismConfig
from slime_plugins.tts_reward import PROTOCOL


def request_json(endpoint, route, payload=None, token=None):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Service endpoint must be an HTTP(S) base URL without query or fragment")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        endpoint.rstrip("/") + route,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"{route} must return a JSON object")
    return result


def teacher_binding(endpoint, checkpoint, auth_env=None):
    token = os.environ.get(auth_env) if auth_env else None
    if auth_env and not token:
        raise ValueError(f"Teacher token environment {auth_env!r} is missing")
    body = validate_admin_result(
        request_json(endpoint, "/model_info", {"stages": ["tts_engine"], "timeout_s": 20}, token),
        "model_info",
    )
    stage = body["results"][0]["data"]
    infos = [stage] + [rank["data"] for rank in stage.get("rank_results", [])]
    candidates = [info for info in infos if info.get("supports_action_scoring") is True]
    if not candidates:
        raise ValueError("The teacher endpoint must run MossTTSPrismScorePipelineConfig")
    info = candidates[0]
    checksum = info.get("teacher_weight_sha256")
    if (
        info.get("supports_weight_update") is not False
        or info.get("stage_tp_size", 1) != 1
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or any(char not in "0123456789abcdef" for char in checksum)
    ):
        raise ValueError("Teacher info must report frozen TP1 weights and their loaded tensor checksum")
    asset = PrismConfig.from_path(checkpoint)
    if info.get("model_identity") != asset.identity:
        raise ValueError("Teacher endpoint identity differs from the supplied local inference artifact")
    for candidate in candidates[1:]:
        if candidate.get("teacher_weight_sha256") != checksum or candidate.get("model_identity") != asset.identity:
            raise ValueError("Teacher stage/rank identities disagree")
    result = {
        "endpoint": endpoint.rstrip("/"),
        "checkpoint": str(Path(checkpoint).resolve()),
        "model_identity": asset.identity,
        "weight_sha256": checksum,
    }
    if auth_env:
        result["auth_token_env"] = auth_env
    return result


def write_config(path, value):
    if path.exists():
        raise FileExistsError(f"Choose a fresh output directory; refusing to overwrite {path}")
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Local Prism inference artifact")
    parser.add_argument("--codec", required=True, help="Local codec artifact or explicit model ID")
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory for concrete YAML files")
    parser.add_argument("--teacher", action="append", nargs=3, default=[], metavar=("NAME", "ENDPOINT", "CHECKPOINT"))
    parser.add_argument("--domain", action="append", nargs=2, default=[], metavar=("DOMAIN", "TEACHER_NAME"))
    parser.add_argument(
        "--deployment",
        action="append",
        nargs=2,
        default=[],
        metavar=("TEACHER_NAME", "DEPLOYMENT_NAME"),
        help="After explicitly inspecting the teacher endpoint, bind it to a frozen named slime deployment",
    )
    parser.add_argument("--teacher-auth-token-env", help="Environment variable read by all configured teacher clients")
    parser.add_argument("--reward", action="append", nargs=3, default=[], metavar=("NAME", "ENDPOINT", "WEIGHT"))
    parser.add_argument("--audio-transport", choices=["path", "upload"], default="path")
    args = parser.parse_args()
    PrismConfig.from_path(args.student)
    models = {}
    for name, endpoint, checkpoint in args.teacher:
        if name in models:
            raise ValueError(f"Duplicate teacher name {name!r}")
        models[name] = teacher_binding(endpoint, checkpoint, args.teacher_auth_token_env)
    deployments = set()
    for name, deployment in args.deployment:
        if name not in models or name in deployments or deployment == "student":
            raise ValueError("Managed binding must name one configured teacher and a frozen non-student deployment")
        deployments.add(name)
        models[name].pop("endpoint")
        models[name]["deployment"] = deployment
    domains = {}
    for domain, name in args.domain:
        if domain in domains or name not in models:
            raise ValueError(f"Duplicate domain or unknown teacher binding: {domain!r} -> {name!r}")
        domains[domain] = name
    if models and not domains:
        raise ValueError("Specify explicit --domain mappings for multi-domain OPD")
    services, weights = {}, {}
    for name, endpoint, weight in args.reward:
        if name not in {"wer", "sim", "judge_clap"} or name in services:
            raise ValueError(f"Unknown or duplicate reward {name!r}")
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Enabled reward weights must be finite and positive")
        info = request_json(endpoint, "/info")
        identity = info.get("identity", {})
        if (
            info.get("protocol") != PROTOCOL
            or not isinstance(identity, dict)
            or not isinstance(identity.get("fingerprint"), str)
            or not identity["fingerprint"].strip()
        ):
            raise ValueError(f"Reward {name!r} does not implement {PROTOCOL} with a frozen identity")
        services[name] = {
            "endpoint": endpoint.rstrip("/"),
            "expected_fingerprint": identity["fingerprint"],
            "audio_transport": args.audio_transport,
            "max_inflight": 2 if name == "judge_clap" else 4,
        }
        weights[name] = weight
    args.output.mkdir(parents=True, exist_ok=True)
    write_config(
        args.output / "student.yaml",
        {
            "config_cls": "MossTTSPrismPipelineConfig",
            "model_path": str(Path(args.student).resolve()),
            "codec_model_path": args.codec,
            "enable_rl": True,
        },
    )
    if models:
        write_config(args.output / "teachers.yaml", {"models": models, "by_domain": domains})
    if services:
        write_config(args.output / "rewards.yaml", {"services": services, "weights": weights})
    print(f"Wrote concrete Prism configs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
