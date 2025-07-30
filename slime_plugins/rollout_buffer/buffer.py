import copy
import glob
import importlib.util
import json
import pathlib
import threading
import time
from typing import Any, Dict, Optional, List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Rollout Buffer Server", debug=True)


def default_is_valid_group(group_data, min_valid_group_size, task_type):
    instance_id, samples = group_data
    return len(samples) >= min_valid_group_size


def default_get_group_data_meta_info(temp_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Default implementation for getting meta information about the temporary data
    collected between get_batch calls.
    """
    if not temp_data:
        return {
            "total_samples": 0,
            "num_groups": 0,
            "avg_group_size": 0,
            "avg_reward": 0,
        }

    meta_info = {"total_samples": 0, "num_groups": len(temp_data)}

    all_rewards = []
    # Calculate per-group statistics
    for instance_id, samples in temp_data.items():
        group_size = len(samples)
        group_rewards = [s["reward"] for s in samples]  # Calculate group reward standard deviation
        meta_info["total_samples"] += group_size
        all_rewards.extend(group_rewards)
    # Calculate global statistics
    meta_info["avg_group_size"] = meta_info["total_samples"] / meta_info["num_groups"]

    if all_rewards:
        meta_info["avg_reward"] = sum(all_rewards) / len(all_rewards)
    else:
        meta_info["avg_reward"] = 0
    return meta_info


def discover_generators():
    """
    Automatically discover generator modules in the generator directory.
    Returns a dictionary mapping task_type to module with run_rollout function.
    """
    generator_map = {}
    generator_dir = pathlib.Path(__file__).parent / "generator"

    # Find all files within generator_dir
    for file_path in glob.glob(str(generator_dir / "*.py")):
        if file_path.endswith("__init__.py"):
            continue

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
            for func_name in [
                "transform_group",
                "is_valid_group",
                "get_group_data_meta_info",
            ]:
                generator_info[func_name] = getattr(module, func_name, None)

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
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
    ):
        self.data = {}
        self.temp_data = {}
        self.group_timestamps = {}
        self.group_size = group_size
        self.task_type = task_type

        # Set up function handlers with defaults
        self.is_valid_group_func = is_valid_group_func or default_is_valid_group
        self.get_group_data_meta_info_func = get_group_data_meta_info_func or default_get_group_data_meta_info
        self.transform_group_func = transform_group_func or (lambda group, task_type: group)

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

    def _get_valid_groups_with_timeout(self, del_data=False):
        """Get valid groups including timeout-based groups"""
        valid_groups = {}
        timed_out_groups = {}
        finished_groups = []

        for instance_id, group_data in self.data.items():
            if self.is_valid_group_func((instance_id, group_data), self.group_size, self.task_type):
                valid_groups[instance_id] = group_data

        # Remove finished groups and timed out groups with insufficient data
        if del_data:
            for instance_id in finished_groups:
                self.data.pop(instance_id, None)
                self.group_timestamps.pop(instance_id, None)
                print(f"Removed finished group {instance_id}")

        # Combine normal valid groups and timeout groups
        all_valid_groups = {**valid_groups, **timed_out_groups}

        return all_valid_groups, finished_groups

    def get(self):
        output = {"data": [], "meta_info": {}}

        # Get meta information about temp data before processing
        meta_info = self.get_group_data_meta_info_func(self.temp_data)
        output["meta_info"] = meta_info

        valid_groups, finished_groups = self._get_valid_groups_with_timeout(del_data=True)
        output["meta_info"]["finished_groups"] = finished_groups

        print(f"meta info: {json.dumps(meta_info, indent=2)}")

        valid_groups = list(valid_groups.items())

        for instance_id, group in valid_groups:
            # First filter individual items
            transformed_group = self.transform_group_func((instance_id, group), self.task_type)
            output["data"].extend(transformed_group[1])

            if instance_id in self.data:
                self.data.pop(instance_id)

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
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
    ):
        self.buffer = BufferQueue(
            group_size=group_size,
            task_type=task_type,
            transform_group_func=transform_group_func,
            is_valid_group_func=is_valid_group_func,
            get_group_data_meta_info_func=get_group_data_meta_info_func,
        )
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.total_written = 0
        self.total_read = 0
        self.task_type = task_type

    def write(self, data):
        with self.lock:
            self.buffer.append(data)
            self.total_written += 1
            self.not_empty.notify_all()
        return data

    def read(self):
        with self.not_empty:
            if len(self.buffer) == 0:
                return {"data": [], "meta_info": {}}

            # Don't clear temp_data for regular read operations
            result = self.buffer.get()
            self.total_read += len(result["data"])
            return result


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


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    items = buffer.read()

    if not items["data"]:
        return BufferResponse(
            success=False,
            message="No data available to read",
            data={"data": [], "meta_info": items["meta_info"]},
        )

    print(f"return {len(items['data'])} items and save them to local")
    buffer.buffer.temp_data = {}

    return BufferResponse(
        success=True,
        message=f"Successfully read {len(items['data'])} items",
        data=items,
    )


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

    buffer = RolloutBuffer(
        group_size=int(data["num_repeat_per_sample"]),
        task_type=task_type,
        transform_group_func=generator_info.get("transform_group", None),
        is_valid_group_func=generator_info.get("is_valid_group"),
        get_group_data_meta_info_func=generator_info.get("get_group_data_meta_info"),
    )

    # Call the run_rollout function from the appropriate generator module
    generator_info["run_rollout"](data)
    print(f"Rollout completed successfully for task_type: {task_type}")


@app.post("/start_rollout")
async def start_rollout(request: Request, background: BackgroundTasks):
    payload = await request.json()
    background.add_task(run_rollout, payload)
    return {"message": "Rollout started"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8889,
        limit_concurrency=1000,  # Connection concurrency limit
        # limit_max_requests=1000000,  # Maximum request limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
