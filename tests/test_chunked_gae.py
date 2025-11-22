import time
import pytest
import torch

from slime.utils.ppo_utils import chunked_gae, vanilla_gae


@pytest.mark.parametrize(
    "B,T",
    [
        (16, 4096),
        (32, 8192),
        (256, 128 * 1024),
    ],
)
@pytest.mark.parametrize("chunk_size", [64, 128, 256])
def test_gae_parallel_matches_serial(B, T, chunk_size):
    """
    Test that chunked_gae (parallel-scan) matches vanilla_gae (batch-serial)
    under various shapes, chunk sizes and dtypes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    rewards = torch.randn(B, T, device=device, dtype=torch.float32)
    values = torch.randn(B, T, device=device, dtype=torch.float32)

    gamma, lam = 0.99, 0.95

    # ---------- Serial ----------
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    adv_s, ret_s = vanilla_gae(rewards, values, gamma, lam)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    serial_time = t1 - t0

    # ---------- Parallel-scan ----------
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    adv_p, ret_p = chunked_gae(rewards, values, gamma, lam, chunk_size=chunk_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    parallel_time = t1 - t0

    # ---------- Accuracy ----------
    adv_err = (adv_s - adv_p).abs().max().item()
    ret_err = (ret_s - ret_p).abs().max().item()

    atol = 1e-5
    assert adv_err < atol, f"adv error too large: {adv_err}"
    assert ret_err < atol, f"ret error too large: {ret_err}"

    # ---------- logging ----------
    print(f"\n[GAE Test] B={B}, T={T}, chunk={chunk_size}")
    print(f"  Serial   : {serial_time:.6f} s")
    print(f"  Parallel : {parallel_time:.6f} s")
    print(f"  Speedup  : x{serial_time / parallel_time:.2f}")
    print(f"  Max diff adv={adv_err:.3e}, ret={ret_err:.3e}")
