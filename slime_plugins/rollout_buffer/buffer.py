# [IMPORTANT] Normalize Process: [raw reward -> normalized reward (only perform on valid reward) -> padded normalized reward]
# Please note we multiply the normalized reward by **group_size / valid_size** to align with GRPO reward
# You can see `normalize_group_data` function at utils.py

import copy
import glob
import importlib.util
import json
import os
import pathlib
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from generator.utils.default_func import (
    default_filter_item,
    default_get_group_data_meta_info,
    default_is_valid_group,
    default_normalize_group_data,
    default_pad_group_data,
)
from pydantic import BaseModel

from tools.visualizer import BufferStatsVisualizer

app = FastAPI(title="Rollout Buffer Server", debug=True)

MAX_SIZE = 1000_000_000


def discover_generators():
    """
    Automatically discover generator modules in the generator directory.
    Returns a dictionary mapping task_type to module with run_rollout function.
    """
    generator_map = {}
    generator_dir = pathlib.Path(__file__).parent / "generator"

    # Find all files ending with _generator.py
    generator_files = glob.glob(str(generator_dir / "*_generator.py"))

    for file_path in generator_files:
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("generator_module", file_path)
            if spec is None or spec.loader is None:
                print(f"Warning: Could not load spec for {file_path}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if module has TASK_TYPE constant
            if not hasattr(module, "TASK_TYPE"):
                print(f"Warning: {file_path} does not define TASK_TYPE constant")
                continue

            # Check if module has run_rollout function
            if not hasattr(module, "run_rollout"):
                print(f"Warning: {file_path} does not define run_rollout function")
                continue

            task_type = getattr(module, "TASK_TYPE")
            generator_info = {
                "module": module,
                "file_path": file_path,
                "run_rollout": getattr(module, "run_rollout"),
            }

            # Check for optional functions and use defaults if not present
            optional_functions = [
                "normalize_group_data",
                "pad_group_data",
                "is_valid_group",
                "get_group_data_meta_info",
                "filter_item",
            ]

            for func_name in optional_functions:
                if hasattr(module, func_name):
                    generator_info[func_name] = getattr(module, func_name)
                    print(f"Found custom {func_name} for {task_type}")
                else:
                    # Use default functions
                    default_func_name = f"default_{func_name}"
                    if default_func_name in globals():
                        generator_info[func_name] = globals()[default_func_name]
                    else:
                        print(f"Warning: No default function found for {func_name}")

            generator_map[task_type] = generator_info
            print(f"Discovered generator: {task_type} -> {file_path}")

        except Exception as e:
            print(f"Error loading generator from {file_path}: {str(e)}")
            continue

    return generator_map


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class BufferQueue:
    def __init__(
        self,
        group_size,
        min_valid_group_size_ratio=1,
        min_valid_item_size_ratio=1,
        max_buffer_size=None,
        task_type="math",
        normalize_group_data_func=None,
        pad_group_data_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
        group_timeout_seconds=300,  # 5 minutes default timeout
        min_timeout_group_size_ratio=0.7,  # minimum ratio for timeout groups (default 70%)
        filter_item_func=None,
    ):
        self.data = {}
        self.temp_data = {}
        self.group_timestamps = {}
        self.group_size = group_size
        self.min_valid_group_size = int(group_size * min_valid_group_size_ratio)
        self.min_valid_item_size = int(group_size * min_valid_item_size_ratio)
        self.max_buffer_size = max_buffer_size
        self.task_type = task_type
        self.group_timeout_seconds = group_timeout_seconds
        self.min_timeout_group_size_ratio = min_timeout_group_size_ratio

        # Set up function handlers with defaults
        self.normalize_group_data = normalize_group_data_func or default_normalize_group_data
        self.pad_group_data_func = pad_group_data_func or default_pad_group_data
        self.is_valid_group_func = is_valid_group_func or default_is_valid_group
        self.get_group_data_meta_info_func = get_group_data_meta_info_func or default_get_group_data_meta_info
        self.filter_item_func = filter_item_func or default_filter_item

    def filter_group_items(self, group_data):
        """
        Filter individual items in a group before normalization.

        Args:
            group_data (tuple): Tuple of (instance_id, items)

        Returns:
            tuple: Filtered (instance_id, items)
        """
        instance_id, items = group_data
        filtered_items = [item for item in items if self.filter_item_func(item, self.task_type)]
        return (instance_id, filtered_items)

    def popleft(self):
        if len(self.data) == 0:
            return None
        return self.data.pop(0)

    def append(self, item):
        instance_id = item["instance_id"]
        current_time = time.time()

        # Update timestamp for this group
        self.group_timestamps[instance_id] = current_time

        if instance_id not in self.temp_data:
            self.temp_data[instance_id] = [copy.deepcopy(item)]
        else:
            self.temp_data[instance_id].append(copy.deepcopy(item))

        if instance_id not in self.data:
            self.data[instance_id] = [item]
        else:
            self.data[instance_id].append(item)

    def _is_group_timed_out(self, instance_id):
        """Check if a group has timed out"""
        if instance_id not in self.group_timestamps:
            return False

        current_time = time.time()
        last_update = self.group_timestamps[instance_id]
        return (current_time - last_update) > self.group_timeout_seconds

    def _get_valid_groups_with_timeout(self, del_data=False):
        """Get valid groups including timeout-based groups"""
        valid_groups = {}
        timed_out_groups = {}
        finished_groups = []

        for instance_id, group_data in self.data.items():
            group_size = len(group_data)
            is_normally_valid, is_finished = self.is_valid_group_func(
                (instance_id, group_data), self.min_valid_group_size, self.task_type
            )
            is_timed_out = self._is_group_timed_out(instance_id)

            # Ensure that valid groups are always finished (valid groups ⊆ finished groups)
            assert not (
                is_normally_valid and not is_finished
            ), f"Group {instance_id} is valid but not finished - this should not happen"

            if is_finished:
                finished_groups.append(instance_id)
                # If finished and valid, include in processing
                if is_normally_valid:
                    valid_groups[instance_id] = group_data

                continue

            if is_timed_out:
                # All timed out groups are considered finished
                finished_groups.append(instance_id)

                actual_ratio = group_size / self.group_size
                if actual_ratio >= self.min_timeout_group_size_ratio:
                    # Timed out but has enough items based on ratio, include for processing
                    timed_out_groups[instance_id] = group_data

        # Remove finished groups and timed out groups with insufficient data
        if del_data:
            for instance_id in finished_groups:
                self.data.pop(instance_id, None)
                self.group_timestamps.pop(instance_id, None)
                print(f"Removed finished group {instance_id}")

        # Combine normal valid groups and timeout groups
        all_valid_groups = {**valid_groups, **timed_out_groups}

        return all_valid_groups, finished_groups

    def get_batch(self, batch_size=1):
        output = {"data": [], "meta_info": {}}

        # Get meta information about temp data before processing
        meta_info = self.get_group_data_meta_info_func(self.temp_data)
        output["meta_info"] = meta_info

        valid_groups, finished_groups = self._get_valid_groups_with_timeout(del_data=True)
        batch_count = sum([len(v) for v in valid_groups.values()])

        output["meta_info"]["finished_groups"] = finished_groups

        print(f"Batch meta info: {json.dumps(meta_info, indent=2)}")
        print(f"Found {len(valid_groups)} valid groups and {len(finished_groups)} finished groups")

        valid_groups = list(valid_groups.items())

        if batch_count < batch_size:
            print(f"Not enough valid data: {batch_count} < {batch_size}")
            return output

        for instance_id, group in valid_groups:
            # First filter individual items
            filtered_group = self.filter_group_items((instance_id, group))
            # Only proceed with normalization if we have enough valid items
            if len(filtered_group[1]) >= self.min_valid_item_size:
                norm_group = self.normalize_group_data(filtered_group)
                pad_group = self.pad_group_data_func(norm_group, self.group_size)
                output["data"].extend(pad_group[1])
            else:
                print(
                    f"instance_id: {instance_id} has {len(filtered_group[1])} items, which is less than {self.min_valid_item_size}"
                )

            if instance_id in self.data:
                self.data.pop(instance_id)
            if len(output["data"]) >= batch_size:
                break

        return output

    def __len__(self):
        valid_groups, _ = self._get_valid_groups_with_timeout()
        num = sum([len(v) for v in valid_groups.values()])
        num_of_all_groups = sum([len(v) for v in self.data.values()])
        print(f"valid_groups: {len(valid_groups)}, num: {num}, num_of_all_groups: {num_of_all_groups}")
        return num


class RolloutBuffer:
    def __init__(
        self,
        group_size=16,
        min_valid_group_size_ratio=1,
        min_valid_item_size_ratio=1,
        max_size=None,
        task_type="math",
        normalize_group_data_func=None,
        pad_group_data_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
        group_timeout_seconds=300,  # 5 minutes default
        min_timeout_group_size_ratio=0.7,  # minimum ratio for timeout groups (default 10%)
        filter_item_func=None,
    ):
        self.buffer = BufferQueue(
            group_size=group_size,
            min_valid_group_size_ratio=min_valid_group_size_ratio,
            min_valid_item_size_ratio=min_valid_item_size_ratio,
            max_buffer_size=max_size,
            task_type=task_type,
            normalize_group_data_func=normalize_group_data_func,
            pad_group_data_func=pad_group_data_func,
            is_valid_group_func=is_valid_group_func,
            get_group_data_meta_info_func=get_group_data_meta_info_func,
            group_timeout_seconds=group_timeout_seconds,
            min_timeout_group_size_ratio=min_timeout_group_size_ratio,
            filter_item_func=filter_item_func,
        )
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.max_size = max_size
        self.total_written = 0
        self.total_read = 0
        self.task_type = task_type

        # Initialize the visualizer
        self.visualizer = BufferStatsVisualizer(time_window=60)  # 60 second window
        # Set args for filename generation
        self.visualizer.set_args(
            {
                "task_type": task_type,
                "group_size": group_size,
                "num_repeat_per_sample": group_size,
            }
        )

        print(
            f"set group_size = {group_size}, timeout = {group_timeout_seconds}s, min_timeout_ratio = {min_timeout_group_size_ratio}"
        )

    def write(self, data):
        with self.not_full:
            while self.max_size and len(self.buffer) >= self.max_size:
                print(f"Buffer is full, waiting for space")
                self.not_full.wait()

            self.buffer.append(data)
            self.total_written += 1

            # Update visualization stats - just increment the counter for current window
            self.visualizer.add_data_point(1)

            self.not_empty.notify_all()
            return data

    def read(self, batch_size=-1, wait=True, timeout=10):
        with self.not_empty:
            if len(self.buffer) < batch_size and wait:
                self.not_empty.wait(timeout=timeout)

            if len(self.buffer) == 0:
                return {"data": [], "meta_info": {}}

            actual_size = min(batch_size, len(self.buffer)) if batch_size != -1 else len(self.buffer)
            # Don't clear temp_data for regular read operations
            result = self.buffer.get_batch(batch_size=actual_size)
            self.total_read += len(result["data"])

            self.not_full.notify_all()
            return result

    def peek(self, batch_size=1):
        with self.lock:
            if len(self.buffer) == 0:
                return {"data": [], "meta_info": {}}
            actual_size = min(batch_size, len(self.buffer))
            # Note: peek doesn't actually get the batch, so we don't get real meta_info
            return {"data": list(self.buffer)[:actual_size], "meta_info": {}}

    def get_stats(self):
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "total_written": self.total_written,
                "total_read": self.total_read,
            }

    def count(self):
        with self.lock:
            return len(self.buffer)

    def close(self):
        """Close the buffer and clean up"""
        if hasattr(self, "visualizer"):
            self.visualizer.close()


buffer = RolloutBuffer()


@app.post("/buffer/write", response_model=BufferResponse)
async def write_to_buffer(request: Request):
    try:
        data = await request.json()
        item = buffer.write(data)
        return BufferResponse(
            success=True,
            message="Data has been successfully written to buffer",
            data={"data": [item], "meta_info": "write to buffer"},
        )
    except Exception as e:
        print(f"Write failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Write failed: {str(e)}")


@app.post("/buffer/read", response_model=BufferResponse)
async def read_from_buffer(request: Request):
    data = await request.json()
    try:
        items = buffer.read(
            batch_size=data["batch_size"],
            timeout=data["timeout"],
        )

        if not items["data"] and data.get("wait", True):
            return BufferResponse(
                success=False,
                message="Timeout waiting, no data available to read",
                data={"data": [], "meta_info": items["meta_info"]},
            )

        return BufferResponse(
            success=True,
            message=f"Successfully read {len(items['data'])} items",
            data=items,  # Return the complete items dictionary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read failed: {str(e)}")


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    data = await request.json()
    current_size = buffer.count()

    if not "batch_size" in data.keys():
        data["batch_size"] = -1

    if data["batch_size"] > 0 and current_size < data["batch_size"]:
        return BufferResponse(
            success=False,
            message=f"Not enough data. Requested {data['batch_size']} items but only {current_size} available.",
            data={"data": [], "meta_info": {}},
        )

    try:
        # Clear temp_data only for get_rollout_data operations
        items = buffer.read(batch_size=data["batch_size"], timeout=600)
    except TimeoutError as e:
        print(f"TimeoutError: {e}")

    if not items["data"]:
        return BufferResponse(
            success=False,
            message="No data available to read",
            data={"data": [], "meta_info": items["meta_info"]},
        )

    if items["data"]:
        print(f"return {len(items['data'])} items and save them to local")
        save_data_to_local(items["data"])
        buffer.buffer.temp_data = {}

    return BufferResponse(
        success=True,
        message=f"Successfully read {len(items['data'])} items",
        data=items,
    )


def save_data_to_local(data):
    try:
        save_dir = "rollout_data"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/rollout_data_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error saving data to local: {str(e)}")


def run_rollout(data: dict):
    global buffer
    # Auto-discover generators
    generator_map = discover_generators()

    task_type = data["task_type"]
    if task_type not in generator_map:
        print(f"Error: No generator found for task_type '{task_type}'")
        print(f"Available generators: {list(generator_map.keys())}")
        return

    generator_info = generator_map[task_type]
    print(f"Using generator: {generator_info['file_path']} for task_type: {task_type}")

    # Extract processing functions from generator
    normalize_func = generator_info.get("normalize_group_data")
    pad_func = generator_info.get("pad_group_data")
    is_valid_func = generator_info.get("is_valid_group")
    get_meta_info_func = generator_info.get("get_group_data_meta_info")
    filter_item_func = generator_info.get("filter_item")

    if "min_valid_group_size_ratio" not in data.keys():
        data["min_valid_group_size_ratio"] = 1

    # Add timeout configuration
    group_timeout_seconds = data.get("group_timeout_seconds", 300)  # 5 minutes default
    min_timeout_group_size_ratio = data.get("min_timeout_group_size_ratio", 0.7)
    if "min_valid_item_size_ratio" not in data.keys():
        data["min_valid_item_size_ratio"] = 0.7

    buffer = RolloutBuffer(
        max_size=MAX_SIZE,
        group_size=int(data["num_repeat_per_sample"]),
        min_valid_group_size_ratio=data["min_valid_group_size_ratio"],
        min_valid_item_size_ratio=data["min_valid_item_size_ratio"],
        min_timeout_group_size_ratio=min_timeout_group_size_ratio,
        task_type=task_type,
        normalize_group_data_func=normalize_func,
        pad_group_data_func=pad_func,
        is_valid_group_func=is_valid_func,
        get_group_data_meta_info_func=get_meta_info_func,
        filter_item_func=filter_item_func,
        group_timeout_seconds=group_timeout_seconds,
    )

    try:
        # Call the run_rollout function from the appropriate generator module
        generator_info["run_rollout"](data)
        print(f"Rollout completed successfully for task_type: {task_type}")
    except Exception as e:
        print(f"Error running rollout for task_type '{task_type}': {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # Save the visualization when rollout is complete
        buffer.close()


@app.post("/start_rollout")
async def start_rollout(request: Request, background: BackgroundTasks):
    payload = await request.json()
    background.add_task(run_rollout, payload)
    return {"message": "Rollout started"}


@app.get("/buffer/peek", response_model=BufferResponse)
async def peek_buffer(request: Request):
    data = await request.json()
    try:
        items = buffer.peek(batch_size=data["batch_size"])
        return BufferResponse(
            success=True,
            message=f"Successfully previewed {len(items['data'])} items",
            data=items,  # Return the complete items dictionary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Peek method failed: {str(e)}")


@app.get("/buffer/stats")
async def get_buffer_stats():
    stats = buffer.get_stats()
    return BufferResponse(
        success=True,
        message="Buffer stats retrieved successfully",
        data={
            "data": [stats],
            "meta_info": {},
        },
    )


@app.get("/")
async def root():
    return {"message": "Rollout Buffer Server is running"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8889,
        limit_concurrency=1000,  # Connection concurrency limit
        # limit_max_requests=1000000,  # Maximum request limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
