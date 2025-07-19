import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from transformers import AutoTokenizer

import wandb
from slime.ray.buffer import Buffer
from slime.utils.async_utils import run
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

__all__ = ["generate_agent_rollout"]


# Global variables for evaluation
TOKENIZER = None
START_ROLLOUT = True


def select_rollout_data(args, results, need_length):
    """
    Select the most recent groups when there are too many samples.
    Groups all samples by instance_id, sorts groups by timestamp.

    Args:
        args: Arguments containing configuration
        results: List of rollout data items with timestamps

    Returns:
        Selected samples from the newest groups based on timestamp cutoff
    """
    if not results:
        return results

    # Group samples by instance_id
    groups = {}
    for item in results:
        assert "instance_id" in item, "instance_id must be in item"
        instance_id = item["instance_id"]
        if instance_id not in groups:
            groups[instance_id] = []
        groups[instance_id].append(item)

    print(f"📊 Total groups: {len(groups)}, total samples: {len(results)}")

    # If we don't have too many samples, return all
    assert need_length < len(results), "need_length must be smaller than results length"

    # Get timestamp for each group (use the latest timestamp in the group)
    def get_group_timestamp(group_items):
        timestamps = []
        for item in group_items:
            if "timestamp" in item:
                timestamps.append(float(item["timestamp"]))
            elif "extra_info" in item and "timestamp" in item["extra_info"]:
                timestamps.append(float(item["extra_info"]["timestamp"]))
        return max(timestamps) if timestamps else 0

    # Create list of (group_id, timestamp, samples) and sort by timestamp
    group_data = []
    for group_id, group_items in groups.items():
        group_timestamp = get_group_timestamp(group_items)
        group_data.append((group_id, group_timestamp, group_items))

    # Sort groups by timestamp (newest first)
    group_data.sort(key=lambda x: x[1], reverse=True)

    selected_groups = group_data[:need_length]

    # Flatten selected groups back to sample list
    selected_results = []
    for group_id, timestamp, group_items in selected_groups:
        selected_results.extend(group_items)

    # Statistics for monitoring
    if selected_groups:
        newest_ts = selected_groups[0][1]
        oldest_ts = selected_groups[-1][1]
        print(f"📈 Selected {len(selected_groups)} groups with {len(selected_results)} samples")
        print(f"📈 Group timestamp range: {oldest_ts:.2f} to {newest_ts:.2f}")
        print(f"📈 Time span: {newest_ts - oldest_ts:.2f} seconds")

    return selected_results


def log_raw_info(args, all_meta_info, rollout_id):
    final_meta_info = {}
    if all_meta_info:
        final_meta_info = {
            "total_samples": sum(meta["total_samples"] for meta in all_meta_info if "total_samples" in meta)
        }

        total_samples = final_meta_info["total_samples"]
        if total_samples > 0:
            weighted_reward_sum = sum(
                meta["avg_reward"] * meta["total_samples"]
                for meta in all_meta_info
                if "avg_reward" in meta and "total_samples" in meta
            )

            final_meta_info.update(
                {
                    "avg_reward": weighted_reward_sum / total_samples,
                }
            )
            if hasattr(args, "use_wandb") and args.use_wandb:
                log_dict = {
                    f"rollout/no_filter/total_samples": final_meta_info["total_samples"],
                    f"rollout/no_filter/avg_reward": final_meta_info["avg_reward"],
                }
                try:
                    if args.use_wandb:
                        log_dict["rollout/step"] = (
                            rollout_id
                            if not args.wandb_always_use_train_step
                            else rollout_id
                            * args.rollout_batch_size
                            * args.n_samples_per_prompt
                            // args.global_batch_size
                        )
                        wandb.log(log_dict)
                    print(f"no filter rollout log {rollout_id}: {log_dict}")
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")
                    print(f"no filter rollout log {rollout_id}: {final_meta_info}")
            else:
                print(f"no filter rollout log {rollout_id}: {final_meta_info}")


async def get_rollout_data(
    api_base_url: str, num: Optional[int] = None, timeout: float = 100.0
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:

    url = f"{api_base_url}/get_rollout_data"
    payload = {}

    if num is not None:
        payload["batch_size"] = num
    print(url)
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response.raise_for_status()
                    resp_json = await response.json()
                    if resp_json["success"]:
                        break
                await asyncio.sleep(3)
                if time.time() - start_time > 30:
                    print("rollout data is not ready, have been waiting for 30 seconds")
                    # Reset start_time to continue waiting or handle timeout differently
                    start_time = time.time()  # Or raise an exception, or return empty list

            data = resp_json["data"]
            meta_info = {}
            if type(data) is list:
                if "data" in data:
                    data = [item["data"] for item in data]
            elif type(data) is dict:
                if "data" in data:
                    meta_info = data["meta_info"]
                    data = data["data"]
            print(f"Meta info: {meta_info}")
            required_keys = {"uid", "instance_id", "messages", "reward", "extra_info"}
            for item in data:
                if not required_keys.issubset(item.keys()):
                    raise ValueError(f"Missing required keys in response item: {item}")

            return data, meta_info

    except aiohttp.ClientError as e:
        print(f"[ERROR] Request failed: {e}")
        raise
    except ValueError as ve:
        # print(f"[ERROR] Invalid data format: {ve}")
        raise
    except asyncio.TimeoutError:
        print(f"[ERROR] Request timed out after {timeout} seconds")
        raise


def start_rollout(api_base_url: str, args, metadata):
    url = f"{api_base_url}/start_rollout"
    print(f"metadata: {metadata}")
    finished_groups_instance_id_list = [item for sublist in metadata.values() for item in sublist]
    payload = {
        "num_process": str(getattr(args, "rollout_num_process", 100)),
        "num_epoch": str(args.num_epoch or 3),
        "remote_engine_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        "remote_buffer_url": args.agent_rollout_buffer_url,
        "task_type": args.rollout_task_type,
        "input_file": args.prompt_data,
        "num_repeat_per_sample": str(args.n_samples_per_prompt),
        "max_tokens": str(args.rollout_max_response_len),
        "sampling_params": {
            "max_tokens": args.rollout_max_response_len,
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
        },
        "tokenizer_path": args.hf_checkpoint,
        "skip_instance_ids": finished_groups_instance_id_list,
    }
    print("start rollout with payload: ", payload)

    while True:
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"[start_rollout] Success: {data}")
            return data
        except Exception as e:
            print(f"[start_rollout] Failed to send rollout config: {e}")


async def generate_agent_rollout(
    args, rollout_id: int, data_buffer: Buffer, evaluation: bool = False
) -> Dict[str, Any]:

    global START_ROLLOUT
    if evaluation:
        raise NotImplementedError("Evaluation rollout is not implemented")

    if START_ROLLOUT:
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.agent_rollout_buffer_url, args, metadata)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False

    data_number_to_fetch = (args.rollout_batch_size - data_buffer.get_buffer_length()) * args.n_samples_per_prompt
    if data_number_to_fetch <= 0:
        print(
            f"❕buffer length: {data_buffer.get_buffer_length()}, buffer has enough data, return {args.rollout_batch_size} prompts"
        )
        return data_buffer.get_samples(args.rollout_batch_size)
    assert (
        data_number_to_fetch % args.n_samples_per_prompt == 0
    ), "data_number_to_fetch must be a multiple of n_samples_per_prompt"
    print(f"INFO: buffer length: {data_buffer.get_buffer_length()}, data_number_to_fetch: {data_number_to_fetch}")
    base_url = args.agent_rollout_buffer_url
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    retry_times = 0
    results = []
    all_meta_info = []

    if args.fetch_trajectory_retry_times == -1:
        print(
            f"⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )
    while args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times:
        try:
            while len(results) < data_number_to_fetch:
                time.sleep(5)
                data, meta_info = await get_rollout_data(api_base_url=base_url)
                results.extend(data)
                if meta_info:
                    all_meta_info.append(meta_info)
                print(f"get rollout data with length: {len(results)}")
            break
        except Exception as err:
            print(f"[get_rollout_data] Failed to get rollout data: {err}, retry times: {retry_times}")
            retry_times += 1

    log_raw_info(args, all_meta_info, rollout_id)

    # Apply group-based data selection if there are too many samples
    results = select_rollout_data(args, results, data_number_to_fetch // args.n_samples_per_prompt)

    if len(all_meta_info) > 0 and "finished_groups" in all_meta_info[0]:
        finished_groups_instance_id_list = []
        for item in all_meta_info:
            finished_groups_instance_id_list.extend(item["finished_groups"])

        data_buffer.update_metadata({str(rollout_id): finished_groups_instance_id_list})

    print("finally get rollout data with length: ", len(results))
    sample_results = []

    for i, record in enumerate(results):
        oai_messages = record["messages"]

        mask_generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)
        token_ids, loss_mask = mask_generator.get_loss_mask(oai_messages)
        response_length = mask_generator.get_response_lengths([loss_mask])[0]

        loss_mask = loss_mask[-response_length:]

        sample_results.append(
            Sample(
                index=record["instance_id"],
                prompt=record["uid"],
                tokens=token_ids,
                response_length=response_length,
                reward=record["reward"],
                status=(
                    Sample.Status.COMPLETED
                    if "finish_reason" not in record["extra_info"] or record["extra_info"]["finish_reason"] != "length"
                    else Sample.Status.TRUNCATED
                ),
                loss_mask=loss_mask,
                metadata={**record["extra_info"], "raw_reward": record["raw_reward"]},
            )
        )
    final_return_results = []

    data_buffer.add_samples(sample_results)
    final_return_results = data_buffer.get_samples(args.rollout_batch_size)

    return final_return_results


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for both training and evaluation."""
    return run(generate_agent_rollout(args, rollout_id, data_buffer, evaluation))
