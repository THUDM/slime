import asyncio
from typing import Union

import aiohttp

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, **kwargs):
    # === NEW: Add comprehensive error handling and validation ===
    try:
        # Debug log
        print(f"[DEBUG] async_rm called for sample. Has response: {sample.response is not None}, "
              f"Has label: {sample.label is not None}")

        if args.custom_rm_path is not None:
            print(f"[DEBUG] Using custom RM: {args.custom_rm_path}")
            rm_function = load_function(args.custom_rm_path)

            # Validate sample before calling reward function
            if sample.response is None:
                print(f"[WARNING] Sample has None response, returning reward=0")
                return 0

            if sample.label is None:
                print(f"[WARNING] Sample has None label, returning reward=0")
                return 0

            print(f"[DEBUG] Calling custom reward function...")
            result = await rm_function(args, sample, **kwargs)
            print(f"[DEBUG] Custom reward function returned: {result} (type: {type(result)})")

            # Validate result
            if result is None:
                print(f"[WARNING] Custom reward function returned None, using 0 instead")
                return 0

            return result

        # Validate sample has required fields
        if sample.response is None:
            print(f"[WARNING] Sample has None response, returning reward=0")
            return 0

        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
        response = sample.response
        label = sample.label

        print(f"[DEBUG] Using built-in RM type: {rm_type}")

        if rm_type.startswith("boxed_"):
            response = extract_boxed_answer(response) or ""
            rm_type = rm_type[len("boxed_") :]

        # This function is intended for remote or time-consuming reward model evaluation.
        # Implement the actual logic as needed.
        if rm_type == "remote_rm":
            return await remote_rm(args, sample)
        elif rm_type == "deepscaler":
            return get_deepscaler_rule_based_reward(response, label)
        elif rm_type == "dapo":
            return compute_score_dapo(response, label)
        elif rm_type == "math":
            return 1 if grade_answer_verl(response, label) else 0
        elif rm_type == "f1":
            return f1_score(response, label)[0]
        elif rm_type == "gpqa":
            return compute_gpqa_reward(response, label, metadata=metadata)
        elif rm_type == "ifbench":
            from .ifbench import compute_ifbench_reward

            return compute_ifbench_reward(response, label, metadata=metadata)
        elif rm_type:
            raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
        else:
            raise NotImplementedError("Rule-based RM type is not specified.")

    except Exception as e:
        print(f"[ERROR] Reward computation failed for sample: {type(e).__name__}: {e}")
        print(f"[ERROR] Sample info: response_len={len(sample.response) if sample.response else 0}, "
              f"label={sample.label}, metadata={sample.metadata}")
        import traceback
        print(f"[ERROR] Traceback:")
        traceback.print_exc()
        # Return 0 instead of None to avoid downstream issues
        return 0


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[Union[int, float]]:
    if args.custom_rm_path is not None:
        # Load the custom reward function
        rm_function = load_function(args.custom_rm_path)

        # === DEBUG: Log what we're about to do ===
        print(f"[DEBUG] batched_async_rm: Processing {len(samples)} samples with custom RM: {args.custom_rm_path}")

        # === NEW: Robust batch/single sample handling ===
        # Try batch mode first (if the function supports it)
        try:
            # Attempt to call with samples list (batch mode)
            print(f"[DEBUG] Attempting batch mode call: rm_function(args, samples[{len(samples)}])")
            result = await rm_function(args, samples, **kwargs)

            # Check if result is valid batch output
            if isinstance(result, list) and len(result) == len(samples):
                print(f"[DEBUG] Batch mode succeeded! Got {len(result)} rewards")
                return result
            else:
                # Result is not a valid batch, fall back to individual processing
                print(f"[WARNING] Custom reward function returned unexpected batch result "
                      f"(expected list of {len(samples)}, got {type(result)} with value: {result}). "
                      f"Falling back to individual sample processing.")
                raise TypeError("Invalid batch result")

        except Exception as e:
            # Function doesn't support batch mode or returned wrong type
            # Fall back to calling it for each sample individually
            print(f"[INFO] Custom reward function does not support batch mode (error: {type(e).__name__}: {e}). "
                  f"Processing {len(samples)} samples individually.")

            # Create fallback coroutine ONCE outside the loop
            async def return_zero():
                """Fallback that returns 0 instead of None to avoid downstream issues"""
                return 0

            tasks = []
            for i, sample in enumerate(samples):
                try:
                    print(f"[DEBUG] Creating task {i+1}/{len(samples)}")
                    tasks.append(rm_function(args, sample, **kwargs))
                except Exception as sample_error:
                    print(f"[ERROR] Failed to create reward task for sample {i}: {sample_error}")
                    import traceback
                    traceback.print_exc()
                    # Use the fallback coroutine that returns 0
                    tasks.append(return_zero())

            try:
                print(f"[DEBUG] Gathering {len(tasks)} reward tasks...")
                rewards = await asyncio.gather(*tasks, return_exceptions=True)
                print(f"[DEBUG] Gather completed. Got {len(rewards)} results")

                # Handle exceptions in results
                cleaned_rewards = []
                none_count = 0
                exception_count = 0

                for i, reward in enumerate(rewards):
                    if isinstance(reward, Exception):
                        print(f"[ERROR] Reward computation failed for sample {i}: {type(reward).__name__}: {reward}")
                        exception_count += 1
                        cleaned_rewards.append(0)  # Use 0 instead of None
                    elif reward is None:
                        print(f"[WARNING] Reward is None for sample {i}, using 0 instead")
                        none_count += 1
                        cleaned_rewards.append(0)  # Use 0 instead of None
                    else:
                        cleaned_rewards.append(reward)

                print(f"[DEBUG] Reward summary: {len(cleaned_rewards)} total, "
                      f"{exception_count} exceptions, {none_count} None values, "
                      f"{len(cleaned_rewards) - exception_count - none_count} valid")

                return cleaned_rewards

            except Exception as gather_error:
                print(f"[ERROR] Failed to gather rewards: {gather_error}")
                import traceback
                traceback.print_exc()
                # Return 0 for all samples if gather fails
                return [0] * len(samples)

    # Default behavior: use built-in reward types
    print(f"[DEBUG] Using built-in reward types for {len(samples)} samples")
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    try:
        rewards = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        cleaned_rewards = []
        for i, reward in enumerate(rewards):
            if isinstance(reward, Exception):
                print(f"[ERROR] Built-in reward computation failed for sample {i}: {reward}")
                cleaned_rewards.append(0)  # Use 0 instead of None
            elif reward is None:
                print(f"[WARNING] Built-in reward is None for sample {i}, using 0 instead")
                cleaned_rewards.append(0)  # Use 0 instead of None
            else:
                cleaned_rewards.append(reward)

        return cleaned_rewards

    except Exception as e:
        print(f"[ERROR] Fatal error in reward computation: {e}")
        import traceback
        traceback.print_exc()
        return [0] * len(samples)  # Use 0 instead of None
