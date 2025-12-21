# Docker release rule

We will publish 2 kinds of docker images:
1. stable version, which based on official sglang release. We will store the patch on those versions.
2. latest version, which aligns to `lmsysorg/sglang:latest`.

current stable version is:
- sglang nightly-dev-20251208-5e2cda61 (5e2cda6158e670e64b926a9985d65826c537ac82), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)

history versions:
- sglang v0.5.5.post1 (303cc957e62384044dfa8e52d7d8af8abe12f0ac), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)

The command to build:

```bash
just release
```

Before each update, we will test the following models with 64xH100:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-355B-A32B sync
