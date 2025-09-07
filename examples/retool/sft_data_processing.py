from datasets import load_dataset

ds = load_dataset("JoeYing/ReTool-SFT")["train"]


def convert(sample):
    conversations = sample["messages"]

    def convert_role(role):
        if role == "user":
            return "user"
        elif role == "assistant":
            return "assistant"
        elif role == "system":
            return "system"
        else:
            raise ValueError(f"Unknown role: {role}")

    messages = [
        {
            "role": convert_role(turn["role"]),
            "content": turn["content"],
        }
        for turn in conversations
    ]

    return {"messages": messages}


ds = ds.map(convert)
ds.to_parquet("./data/retool/ReTool-SFT.parquet")
