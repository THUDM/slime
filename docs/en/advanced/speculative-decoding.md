# Speculative Decoding

Speculative decoding is a key optimization for speeding up rollouts. Instead of having the expensive target model decode token by token during inference, a lightweight draft model first decodes ahead to produce several tokens, and then the target model verifies them in a batch.

## Accelerating Inference with Speculative Decoding

For models with MTP layers (e.g., GLM-4.6, DeepSeek-V3/R1), simply add:

```bash
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

If you want to use a separately trained draft model (e.g., one trained with [SpecForge](https://docs.sglang.ai/SpecForge/)), also set:

```bash
--sglang-speculative-draft-model-path /your/draft/model/path
```

For detailed parameter meanings and configuration, see SGLang’s speculative decoding [documentation](https://docs.sglang.ai/advanced_features/speculative_decoding.html).

## Online SFT for the Draft Model

As RL progresses, the sampling distributions of the draft and target models can drift apart. Fewer draft tokens pass verification, and speculative decoding can even yield negative returns.

slime currently supports online training of the MTP layers during RL, updating the draft model in sync with training to consistently improve sampling speed. See the related rationale in this [blog](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e). Use it as follows:

```bash
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

And note that this requires a torch dist checkpoint with the MTP weight, you need to add `--mtp-num-layers 1` during the checkpoint conversion from huggingface to torch dist.

Training external draft models is still a WIP.
