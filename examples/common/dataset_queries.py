#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

SLIME_ROOT = Path(__file__).resolve().parents[2]
if str(SLIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SLIME_ROOT))

from examples.common.dataset_registry import default_train_datasets_for_group
from examples.common.dataset_selection import (
    domain_signature,
    group_signature_for_train_datasets,
    resolve_eval_datasets,
)


def _split_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def command_default_train_datasets_for_group(args: argparse.Namespace) -> int:
    print(",".join(default_train_datasets_for_group(args.group)))
    return 0


def command_group_signature(args: argparse.Namespace) -> int:
    dataset_names = _split_csv(args.datasets) + _split_csv(args.dataset_extras)
    path_names = _split_csv(args.paths) + _split_csv(args.path_extras)
    if dataset_names:
        print(group_signature_for_train_datasets(dataset_names))
    elif path_names:
        print("custom")
    else:
        print("unknown")
    return 0


def command_domain_signature(args: argparse.Namespace) -> int:
    print(domain_signature(_split_csv(args.domains)))
    return 0


def command_resolve_eval_datasets(args: argparse.Namespace) -> int:
    resolved = resolve_eval_datasets(
        pool_root=Path(args.pool_root),
        datasets=_split_csv(args.datasets) or None,
        dataset_extras=_split_csv(args.dataset_extras) or None,
        paths=_split_csv(args.paths) or None,
        path_extras=_split_csv(args.path_extras) or None,
    )
    for item in resolved:
        print(f"{item.name}\t{item.source}\t{item.n_samples_per_eval_prompt}")
    return 0


def command_load_eval_config(args: argparse.Namespace) -> int:
    with open(args.path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for dataset in (cfg.get("eval") or {}).get("datasets", []):
        print(f"{dataset['name']}:{dataset['path']}")
    return 0


def command_load_json_manifest(args: argparse.Namespace) -> int:
    with open(args.path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for dataset in manifest.get("datasets", []):
        print(f"{dataset['name']}:{dataset['path']}")
    return 0


def _add_common_csv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", default="")
    parser.add_argument("--dataset-extras", default="")
    parser.add_argument("--paths", default="")
    parser.add_argument("--path-extras", default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    group_defaults = subparsers.add_parser("default-train-datasets-for-group")
    group_defaults.add_argument("--group", required=True)
    group_defaults.set_defaults(func=command_default_train_datasets_for_group)

    group_sig = subparsers.add_parser("group-signature")
    _add_common_csv_args(group_sig)
    group_sig.set_defaults(func=command_group_signature)

    domain_sig = subparsers.add_parser("domain-signature")
    domain_sig.add_argument("--domains", default="")
    domain_sig.set_defaults(func=command_domain_signature)

    resolve_eval = subparsers.add_parser("resolve-eval-datasets")
    resolve_eval.add_argument("--pool-root", required=True)
    _add_common_csv_args(resolve_eval)
    resolve_eval.set_defaults(func=command_resolve_eval_datasets)

    load_eval = subparsers.add_parser("load-eval-config")
    load_eval.add_argument("--path", required=True)
    load_eval.set_defaults(func=command_load_eval_config)

    load_json_manifest = subparsers.add_parser("load-json-manifest")
    load_json_manifest.add_argument("--path", required=True)
    load_json_manifest.set_defaults(func=command_load_json_manifest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
