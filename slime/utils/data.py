import random

from datasets import Dataset as hf_ds

from slime.utils.types import Sample

__all__ = ["Dataset"]


# TODO: don't read the whole file into memory.
def read_file(path):
    if path.endswith(".jsonl") or path.endswith(".json"):
        ds = hf_ds.from_json(path)
    elif path.endswith(".parquet"):
        ds = hf_ds.from_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    for data in ds:
        yield data


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        max_length,
        *,
        prompt_key="text",
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
    ):
        self.origin_samples = []
        for data in read_file(path):
            prompt = data[prompt_key]
            if apply_chat_template:
                if tool_key is not None:
                    tools = data[tool_key]
                else:
                    tools = None
                prompt = tokenizer.apply_chat_template(prompt, tools, tokenize=False, add_generation_prompt=True)

            # TODO: this is slow.
            if max_length is not None:
                if len(tokenizer(prompt)["input_ids"]) > max_length:
                    continue

            self.origin_samples.append(
                Sample(
                    prompt=prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=data.get(metadata_key) or {},
                )
            )

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
