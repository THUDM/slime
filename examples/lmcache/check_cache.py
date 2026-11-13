"""Check a cold request or external KV reuse after a same-checkpoint SGLang restart."""

import argparse
import json
from urllib.request import Request, urlopen


def check_cache(base_url: str, phase: str) -> None:
    request = Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(
            {
                "text": "The quick brown fox jumps over the lazy dog. " * 256,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)

    meta = result["meta_info"]
    cached = meta["cached_tokens"]
    assert isinstance(cached, int) and 0 <= cached <= meta["prompt_tokens"], meta
    if phase == "cold":
        assert cached == 0, f"Expected an empty cache; start a fresh LMCache server: {meta}"
    else:
        details = meta["cached_tokens_details"]
        assert cached > 0 and details["host"] > 0, f"Expected external KV reuse: {meta}"
        assert details["device"] == 0, f"Local radix hit; restart SGLang before this check: {meta}"
    print(json.dumps({"phase": phase, "meta_info": meta, "text": result["text"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Direct SGLang engine URL, without /v1")
    parser.add_argument("phase", choices=("cold", "warm"))
    args = parser.parse_args()
    check_cache(args.base_url, args.phase)
