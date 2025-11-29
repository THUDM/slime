import asyncio
import json
from typing import Annotated

import typer
from openai import AsyncOpenAI

from slime.utils.data import read_file


# can unify w/ sglang_rollout.py later, e.g. add RM, if needed
def main(
    prompt_data: Annotated[str, typer.Option()],
    url: Annotated[str, typer.Option()] = "http://localhost:30000/v1",
    input_key: Annotated[str, typer.Option()] = "input",
    n_samples_per_prompt: Annotated[int, typer.Option()] = 1,
    rollout_max_response_len: Annotated[int, typer.Option()] = 1024,
    rollout_temperature: Annotated[float, typer.Option()] = 1.0,
    rollout_top_p: Annotated[float, typer.Option()] = 1.0,
):
    """
    Minimally send prompts to SGLang using OpenAI endpoints with arguments in the same format as main Slime.

    Example usage:
    python -m slime.utils.debug_utils.send_to_sglang --prompt-data /root/datasets/aime-2024/aime-2024.jsonl --input-key prompt --n-samples-per-prompt 16 --rollout-max-response-len 32768 --rollout-temperature 0.8 --rollout-top-p 0.7
    """

    async def _main_async():
        tasks = [
            asyncio.create_task(_run_one(row, row_index=row_index, repeat_index=repeat_index))
            for row_index, row in enumerate(read_file(prompt_data))
            for repeat_index in range(n_samples_per_prompt)
        ]
        outputs = await asyncio.gather(*tasks)
        for output in outputs:
            print(json.dumps(output))

    async def _run_one(row, row_index: int, repeat_index: int):
        resp = await client.chat.completions.create(
            messages=row[input_key],
            model="dummy_model",
            max_tokens=rollout_max_response_len,
            temperature=rollout_temperature,
            top_p=rollout_top_p,
        )
        return dict(
            row_index=row_index,
            repeat_index=repeat_index,
            **row,
            response=resp.choices[0].message.content,
        )

    client = AsyncOpenAI(api_key="dummy_key", base_url=url)
    asyncio.run(_main_async())


if __name__ == "__main__":
    typer.run(main)
