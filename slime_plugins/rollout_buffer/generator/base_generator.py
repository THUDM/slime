import copy
import json
import random
import time
import uuid
from functools import partial
from multiprocessing import Process, Queue
from time import sleep
from typing import List, Optional

import requests
from openai import OpenAI
from tqdm import tqdm
from slime.rollout.rm_hub import get_deepscaler_rule_based_reward

TASK_TYPE = "math"

SAMPLING_PARAMS = {
    "top_p": 1,
}


def get_rule_based_math_reward(item):
    messages = item["messages"]
    label = item["label"]
    assert messages[-1]["role"] == "assistant", "last message must be assistant, but got {}".format(
        messages[-1]["role"]
    )

    response = messages[-1]["content"]
    if response is None or len(response) == 0:
        return 0

    reward = get_deepscaler_rule_based_reward(response, label)
    return reward


def query_single_turn(client, messages, sampling_params, tools=None):
    base_payload = {
        "messages": messages,
        **sampling_params,
        "model": "custom",
        "stream": False,
        "seed": random.randint(1, 10000000),
        "tools": tools,
    }

    text = None
    accumulated_tokens = 0
    finish_reason = "stop"

    for attempt in range(6):
        try:
            # Create a fresh payload for each attempt
            current_payload = copy.deepcopy(base_payload)

            if text is not None:
                # Update messages with current progress
                current_messages = copy.deepcopy(messages)
                current_messages.append({"role": "assistant", "content": text})
                current_payload["messages"] = current_messages

                # Adjust max_tokens based on accumulated tokens
                if "max_tokens" in sampling_params:
                    current_payload["max_tokens"] = max(0, sampling_params["max_tokens"] - accumulated_tokens)

                # Add continue flag for partial rollouts
                current_payload["extra_body"] = {"continue_final_message": True}
            if current_payload["max_tokens"] == 0:
                break
            response = client.chat.completions.create(**current_payload)

            if len(response.choices) > 0:
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "abort":
                    print(
                        f"query failed, reason: {response.choices[0].finish_reason}, currently generated: {response.usage.completion_tokens}"
                    )

                    accumulated_tokens += response.usage.completion_tokens

                    if text is None:
                        text = response.choices[0].message.content
                    else:
                        text += response.choices[0].message.content

                    sleep(10)
                    continue
                if text is None:
                    text = response.choices[0].message.content
                elif response.choices[0].message.content is not None:
                    text += response.choices[0].message.content
                break
            else:
                print(f"Error in query, status code: {response.status_code}")
                continue
        except Exception as e:
            print(f"query failed in single turn, error: {e}")
            continue

    # Update final messages
    if len(messages) > 0 and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    messages.append({"role": "assistant", "content": text})

    return messages, finish_reason


def worker_process(task_queue, done_queue, rollout_func, reward_func, client, sampling_params):

    for line in iter(task_queue.get, "STOP"):
        if isinstance(line, str):
            item = json.loads(line)
        else:
            item = line

        # try:
        messages, finish_reason = rollout_func(client, item["prompt"], sampling_params)

        item["uid"] = str(uuid.uuid4())
        item["messages"] = messages
        reward = reward_func(item)
        item["rollout_index"] = 1
        item["reward"] = reward
        item["extra_info"] = {}
        item.update(sampling_params)
        item["timestamp"] = str(time.time())
        item["round_number"] = len([_ for _ in item["messages"] if _["role"] == "assistant"])
        item["finish_reason"] = finish_reason

        output_item = {
            "uid": item.pop("uid"),
            "messages": messages,
            "reward": reward,
            "instance_id": item.pop("instance_id"),
            "extra_info": item,
        }

        done_queue.put(output_item)

    done_queue.put("COMPLETE")


class BaseGenerator:
    def __init__(
        self,
        remote_engine_url,
        remote_buffer_url,
        num_repeat_per_sample=1,
        queue_size=1000000,
        num_process=10,
        task_type="math",
        max_tokens=4096,
        num_repeats=10,
        skip_instance_ids: Optional[List[str]] = None,
    ):
        self.queue_size = queue_size
        self.num_process = num_process
        self.remote_engine_url = remote_engine_url
        self.remote_buffer_url = remote_buffer_url
        self.num_repeat_per_sample = num_repeat_per_sample
        self.task_type = task_type
        self.max_tokens = max_tokens
        self.num_repeats = num_repeats
        # Ensure skip_instance_ids is a mutable list (copy to avoid modifying original)
        self.skip_instance_ids = list(skip_instance_ids) if skip_instance_ids is not None else None

        if self.skip_instance_ids is not None:
            print(f"BaseGenerator initialized with {len(self.skip_instance_ids)} instance_ids to skip")
            self.skip_instance_ids = self.skip_instance_ids * self.num_repeat_per_sample

        if "/v1" in remote_engine_url:
            self.client = OpenAI(api_key="test", base_url=remote_engine_url)
        else:
            remote_engine_url = remote_engine_url.strip("/") + "/v1"
            self.client = OpenAI(api_key="test", base_url=remote_engine_url)

    def send_data_to_buffer(self, data):
        remote_buffer_url = self.remote_buffer_url.rstrip("/") + "/buffer/write"

        for _ in range(2):
            try:
                response = requests.post(remote_buffer_url, json=data)
                if response.status_code == 200:
                    break
                else:
                    print(f"send data to buffer failed, status code: {response.status_code}")
                    continue
            except Exception as e:
                print(f"send data to buffer failed, error: {e}")
                continue

    def run(self, input_file, rollout_func, reward_func):
        task_queue, done_queue = Queue(maxsize=self.queue_size), Queue(maxsize=self.queue_size)

        def read_data_into_queue():
            cnt = 0
            items = []
            skipped_count = 0
            with open(input_file, "r") as f:
                for i, line in enumerate(f):
                    item = json.loads(line)
                    if "instance_id" not in item:
                        item["instance_id"] = i
                    items.append(item)
            random.shuffle(items)

            for _ in range(self.num_repeats):

                for item in items:
                    for _ in range(self.num_repeat_per_sample):
                        item_repeat = copy.deepcopy(item)

                        if "uid" not in item_repeat:
                            item_repeat["uid"] = str(uuid.uuid4())

                        # Check if instance_id should be skipped
                        if self.skip_instance_ids is not None and item_repeat["instance_id"] in self.skip_instance_ids:
                            print(f"Skipping instance_id: {item_repeat['instance_id']}")
                            # Remove from skip list to handle potential duplicates in multiple epochs
                            self.skip_instance_ids.remove(item_repeat["instance_id"])
                            skipped_count += 1
                            continue

                        task_queue.put(item_repeat)
                    cnt += 1
                time.sleep(300)

            if skipped_count > 0:
                remaining_skip_count = len(self.skip_instance_ids) if self.skip_instance_ids is not None else 0
                print(
                    f"Rollout summary: skipped {skipped_count} instance_ids, {remaining_skip_count} still in skip list"
                )

            for _ in range(self.num_process):
                task_queue.put("STOP")

        processes = []
        SAMPLING_PARAMS["max_tokens"] = self.max_tokens

        for _ in range(self.num_process):
            process = Process(
                target=partial(worker_process, client=self.client, sampling_params=SAMPLING_PARAMS),
                args=(task_queue, done_queue, rollout_func, reward_func),
            )
            process.start()
            processes.append(process)

        process = Process(target=read_data_into_queue)
        process.start()

        progress_bar = tqdm()
        num_finished = 0
        while num_finished < self.num_process:
            item = done_queue.get()
            if item == "COMPLETE":
                num_finished += 1
            else:
                assert "reward" in item, f"reward not in item: {item}"
                assert "instance_id" in item, f"instance_id not in item: {item}"
                self.send_data_to_buffer(item)
                progress_bar.update(1)

        progress_bar.close()

        return "finished"

    def entry(self, input_file, rollout_func, reward_func, num_epoch=1):
        for _ in range(num_epoch):
            status = self.run(input_file, rollout_func, reward_func)


def run_rollout(data: dict):

    print(f"Starting math rollout with data: {data}")

    rollout_func = query_single_turn
    reward_func = get_rule_based_math_reward

    print(f"Waiting for 10 seconds for buffer server to start")
    time.sleep(10)
    global SAMPLING_PARAMS
    for k, v in data["sampling_params"].items():
        SAMPLING_PARAMS[k] = v
        print(f"Set {k} to {v}", type(v))

    generator = BaseGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        num_repeat_per_sample=int(data["num_repeat_per_sample"]),
        queue_size=1000000,
        max_tokens=int(data["sampling_params"]["max_tokens"]),
        num_process=int(data.get("num_process", 100)),
        task_type=data["task_type"],
        skip_instance_ids=data.get("skip_instance_ids", None),
    )

    generator.entry(data["input_file"], rollout_func, reward_func, int(data.get("num_epoch", 1)))


def normalize_group_data(group, epsilon=1e-8, algo="grpo"):
    print(f"Using math-specific normalization for group {group[0]}")

    assert algo == "grpo", "Only 'grpo' is supported for now."

    instance_id = group[0]
    data = group[1]
    rewards = [item["reward"] for item in data]

    valid_rewards = [r for r in rewards if 1 >= r >= 0]

    if set(valid_rewards) == {0}:
        normalized_rewards = rewards
    else:
        mean_reward = sum(valid_rewards) / len(valid_rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in valid_rewards) / len(valid_rewards)) ** 0.5

        if std_reward < epsilon:
            print(f"[Math Info] Zero variance in group {instance_id}, setting all to 0.")
            normalized_rewards = [0.0 if 1 >= r >= 0 else r for r in rewards]
        else:
            normalized_rewards = [(r - mean_reward) / (std_reward + epsilon) if 1 >= r >= 0 else r for r in rewards]

    for i, item in enumerate(data):
        item["reward"] = normalized_rewards[i]
        item["raw_reward"] = rewards[i]

    return (instance_id, data)


def is_valid_group(group, min_valid_group_size, task_type="math"):
    # Handle both tuple and list inputs
    if isinstance(group, tuple):
        instance_id, items = group
    else:
        items = group

    # Count valid items (non-empty responses)
    valid_indices = []
    for i, item in enumerate(items):
        if item["messages"][-1]["content"].strip():
            valid_indices.append(i)

    group_size = len(items)
    valid_count = len(valid_indices)

    # A group is finished if it has reached the target size
    is_finished = group_size >= min_valid_group_size

    is_valid = is_finished and valid_count >= min_valid_group_size

    return is_valid
