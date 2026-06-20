from datasets import load_dataset

# Load the original dataset
ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")


# Map to extract the ground_truth from the reward_model dict and create a new 'label' field
def transform(example):
    return {
        "prompt": example["prompt"][0]["content"] if example["prompt"] else None,
        "label": example["reward_model"]["ground_truth"],
    }


ds2 = ds.map(transform, remove_columns=ds.column_names)

# Optionally, verify the first few entries
print(ds2[0])

# save to jsonl
ds2.to_json("/root/dapo-math-17k-processed/dapo_math_17k_cleaned.jsonl", orient="records", lines=True)
