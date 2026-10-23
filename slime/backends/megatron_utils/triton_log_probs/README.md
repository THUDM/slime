# Vendored verl linear cross entropy kernel

Source: `verl` commit `4905d0cf4ebc7297231e15efa4cf837163efca45`.

| Source file | SHA-256 before Slime-specific import changes |
| --- | --- |
| `verl/utils/kernel/kernels.py` | `6d8dda8032ffd3caeb8fb2b6c3307126e0892135aee719f69dd26febc76c012b` |
| `verl/utils/kernel/linear_cross_entropy.py` | `b608f75336e18faef233c3d37fe4763a6b08605d92cd977a130bee013ed2e12a` |

The kernel files retain their Apache-2.0 notices. `kernels.py` replaces verl
device helpers with the equivalent `torch.cuda` APIs and applies Slime's BF16
output-boundary compatibility patch: tiled GEMM logits are rounded to BF16,
converted back to FP32, then temperature and softmax statistics run in FP32.
The autograd wrapper additionally makes upstream gradients contiguous because
Slime packs log probabilities and entropy into one pipeline tensor.
