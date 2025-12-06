import asyncio
from typing import Annotated

import ray
import torch
import typer

from slime.utils.misc import load_function
from slime.utils.types import Sample


def _truncate(text, max_len=200):
    """Truncate text and add ellipsis if too long."""
    if text is None:
        return None
    text = str(text).replace("\n", "\\n")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def main(
    rollout_data_path: Annotated[str, typer.Option()],
    custom_rm_path: Annotated[str, typer.Option()],
):
    if not ray.is_initialized():
        ray.init()

    pack = torch.load(rollout_data_path)
    samples = [Sample.from_dict(s) for s in pack["samples"]]
    asyncio.run(_main_async(samples=samples, custom_rm_path=custom_rm_path))


async def _main_async(samples, custom_rm_path):
    rm_function = load_function(custom_rm_path)
    rewards = await asyncio.gather(*[rm_function(None, sample) for sample in samples])

    for i, (sample, reward) in enumerate(zip(samples, rewards, strict=True)):
        print("-" * 60)
        print(f"Sample {i + 1}/{len(samples)}")
        print(f"  Index:    {sample.index}")
        print(f"  Status:   {sample.status}")
        print(f"  Reward:   {reward}")
        print(f"  Prompt:   {_truncate(sample.prompt, 200)}")
        print(f"  Response: {_truncate(sample.response, 200)}")
    print("-" * 60)


if __name__ == "__main__":
    typer.run(main)
