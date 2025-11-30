import asyncio
import time
from typing import Any, Dict, List

import wandb
import aiohttp
import requests


from slime.utils.async_utils import run
from slime.utils.types import Sample

__all__ = ["generate_rollout"]


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
            elif "metadata" in item and "timestamp" in item["metadata"]:
                timestamps.append(float(item["metadata"]["timestamp"]))
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
        selected_results.append(group_items)

    # Statistics for monitoring
    if selected_groups:
        newest_ts = selected_groups[0][1]
        oldest_ts = selected_groups[-1][1]
        print(
            f"📈 Selected {len(selected_groups)} groups with {len(selected_results)*args.n_samples_per_prompt} samples"
        )
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
                    step = (
                        rollout_id
                        if not args.wandb_always_use_train_step
                        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
                    )
                    if args.use_wandb:
                        log_dict["rollout/step"] = step
                        wandb.log(log_dict)

                    if args.use_tensorboard:
                        from slime.utils.tensorboard_utils import _TensorboardAdapter

                        tb = _TensorboardAdapter(args)
                        tb.log(data=log_dict, step=step)
                    print(f"no filter rollout log {rollout_id}: {log_dict}")
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")
                    print(f"no filter rollout log {rollout_id}: {final_meta_info}")
            else:
                print(f"no filter rollout log {rollout_id}: {final_meta_info}")


async def get_rollout_data(api_base_url: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.post(
                f"{api_base_url}/get_rollout_data", json={}, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
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
        if isinstance(data, list):
            if "data" in data[0]:
                data = [item["data"] for item in data]
        elif isinstance(data, dict):
            if "data" in data:
                meta_info = data["meta_info"]
                data = data["data"]
        print(f"Meta info: {meta_info}", flush=True)

        required_keys = {"instance_id", "reward", "metadata"}
        for item in data:
            if not required_keys.issubset(item.keys()):
                raise ValueError(f"Missing required keys in response item: {item}")

        return data, meta_info


def start_rollout(api_base_url: str, args, metadata):
    url = f"{api_base_url}/start_rollout"
    print(f"metadata: {metadata}")
    finished_groups_instance_id_list = [item for sublist in metadata.values() for item in sublist]
    payload = {
        "args":{
            "rollout_temperature": args.rollout_temperature,
            "rollout_top_p": args.rollout_top_p,
            "rollout_top_k": args.rollout_top_k,
            "rollout_max_response_len": args.rollout_max_response_len,
            "rollout_stop": args.rollout_stop,
            "rollout_stop_token_ids": args.rollout_stop_token_ids,
            "rollout_skip_special_tokens": args.rollout_skip_special_tokens,
            "hf_checkpoint": args.hf_checkpoint,
            "prompt_data": args.prompt_data,
            "rollout_max_prompt_len": args.rollout_max_prompt_len,
            "input_key": args.input_key,
            "label_key": args.label_key,
            "metadata_key": args.metadata_key,
            "tool_key": args.tool_key,
            "apply_chat_template": args.apply_chat_template,
            "rollout_seed": args.rollout_seed
        },
        "num_process": str(getattr(args, "rollout_num_process", 128)),
        "num_epochs": str(getattr(args, "num_epoch", None) or 10),
        "remote_engine_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        "remote_buffer_url": args.rollout_buffer_url,
        "task_type": args.rollout_task_type,
        "input_file": args.prompt_data,
        "num_repeat_per_sample": args.n_samples_per_prompt,
        "abort_sleep_time": getattr(args, "abort_sleep_time", 30),
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


async def generate_rollout_async(args, rollout_id: int, data_buffer, evaluation: bool = False) -> Dict[str, Any]:

    global START_ROLLOUT
    if evaluation:
        raise NotImplementedError("Evaluation rollout is not implemented")

    if START_ROLLOUT:
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.rollout_buffer_url, args, metadata)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False

    data_number_to_fetch = args.rollout_batch_size * args.n_samples_per_prompt - data_buffer.get_buffer_length()
    if data_number_to_fetch <= 0:
        print(
            f"❕buffer length: {data_buffer.get_buffer_length()}, buffer has enough data, return {args.rollout_batch_size} prompts"
        )
        return data_buffer.get_samples(args.rollout_batch_size)
    assert (
        data_number_to_fetch % args.n_samples_per_prompt == 0
    ), "data_number_to_fetch must be a multiple of n_samples_per_prompt"
    
    print(f"INFO: buffer length: {data_buffer.get_buffer_length()}, data_number_to_fetch: {data_number_to_fetch}", flush=True)

    retry_times = 0
    results = []
    all_meta_info = []

    # 增加 drop_first_buffer_data 标志
    first_fetch = True

    if args.fetch_trajectory_retry_times == -1:
        print(
            f"⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )
    while args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times:
        try:
            while len(results) < data_number_to_fetch:
                time.sleep(3)
                data, meta_info = await get_rollout_data(api_base_url=args.rollout_buffer_url)

                # 如果第一次取数据并且启用了drop_first_buffer_data，则丢弃
                if first_fetch and getattr(args, "drop_first_buffer_data", False):
                    print("⚠️ Dropping first batch of rollout data due to drop_first_buffer_data=True")
                    first_fetch = False
                    continue  # 直接跳过此次循环，不添加数据

                results.extend(data)
                first_fetch = False
                
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

    print(f"finally get rollout data with length: {len(results)}", flush=True)
    sample_results = []

    for group_record in results:
        group_results = []
        for record in group_record:
            instance_id = record.pop("instance_id")
            group_results.append(Sample.from_dict(record))
        sample_results.append(group_results)

    print(
        f"prompt:{repr(sample_results[-1][0].prompt)}\n"
        f"response:{repr(sample_results[-1][0].response)}\n"
        f"label:{sample_results[-1][0].label}\n"
        f"reward:{sample_results[-1][0].reward}\n",
        flush=True
    )

    data_buffer.add_samples(sample_results)
    final_return_results = data_buffer.get_samples(args.rollout_batch_size)  # type: ignore
    
    return final_return_results


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for both training and evaluation."""
    return run(generate_rollout_async(args, rollout_id, data_buffer, evaluation))
