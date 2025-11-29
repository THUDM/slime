import copy
import json
import time
import uuid
from functools import partial
from multiprocessing import Process, Queue
from types import SimpleNamespace
from typing import List, Optional
import requests

from pebble import ThreadPool
from tqdm import tqdm
from transformers import AutoTokenizer


from slime.utils.types import Sample
from slime.utils.data import Dataset
from slime.rollout.rm_hub import get_deepscaler_rule_based_reward

TASK_TYPE = "math"


def post_sync(url, payload, timeout=6000):
    """同步版本的 post，替换原来的 async post"""
    # 这里只是保留参数接口，use_http2 在 requests 中无效，因为它只支持 HTTP/1.1
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        output = resp.json()
        return output  # 成功直接返回
    except Exception as e:
        # print(f"[post_sync] Error: {e}, retrying... ({retry_count}/{max_retries})", flush=True)
        # time.sleep(30)  # 同步等待
        return None


def get_rule_based_math_reward(item):
    response, label = item["response"], item["label"]    
    if response is None or len(response) == 0:
        return 0
    reward = get_deepscaler_rule_based_reward(response, label)
    return reward


def math_rollout_func(remote_engine_url, item, sampling_params, data_dict: dict):
    instance_id = item.pop("instance_id")
    sample = Sample.from_dict(item)

    url = f"{remote_engine_url}/generate"
    abort_count = 0
    abort_sleep_time = data_dict.get("abort_sleep_time", 30)

    for attempt in range(8):
        payload = {
            "input_ids": sample.tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        output = post_sync(url, payload)
        finish_type = "timeout" if output is None else output["meta_info"]["finish_reason"]["type"]

        if finish_type == "abort":
            abort_count += 1
            sample.status = Sample.Status.ABORTED
            print(f"[math_rollout_func] Abort detected in {attempt} attempt, waiting {abort_sleep_time}s before retry ({attempt}/{abort_count})...")
            time.sleep(abort_sleep_time)
            continue
        elif finish_type == "timeout":
            print(f"[math_rollout_func] timeout detected in {attempt} attempt, waiting {abort_sleep_time}s before attempt {attempt} ...")
            time.sleep(abort_sleep_time)
            continue

        # 如果不是 abort，就更新 sample 的输出信息
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [tok[1] for tok in output["meta_info"]["output_token_logprobs"]]
            new_response_log_probs = [tok[0] for tok in output["meta_info"]["output_token_logprobs"]]
        else:
            new_response_tokens = []
            new_response_log_probs = []

        sample.tokens.extend(new_response_tokens)
        sample.response += output.get("text", "")
        sample.response_length += len(new_response_tokens)
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs.extend(new_response_log_probs)

        # 设置状态并直接返回
        match finish_type:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "stop":
                sample.status = Sample.Status.COMPLETED
            case _:
                # 未知类型也直接返回
                sample.status = Sample.Status.COMPLETED

        return sample.to_dict(), instance_id

    return sample.to_dict(), instance_id

def worker_process(task_queue, done_queue, rollout_func, reward_func, remote_engine_url=None, sampling_params=None, data_dict=None):
    """极简版 - 无序输出版本"""

    def _process_single_task(data, rollout_func, reward_func, remote_engine_url=None, sampling_params=None, data_dict=None):
        """处理单个任务"""
        if isinstance(data, str):
            item = json.loads(data)
        else:
            item = data
        
        sample_dict, instance_id = rollout_func(remote_engine_url, item, sampling_params, data_dict)
        reward = reward_func(sample_dict)
        sample_dict["reward"] = reward
        sample_dict["instance_id"] = instance_id
        sample_dict["metadata"] = {"timestamp": str(time.time())}
        
        return sample_dict
    
    with ThreadPool(max_workers=data_dict.get("num_repeat_per_sample", 32) * 8) as pool:
        active_futures = set()
        
        # 初始化：提交初始任务
        for _ in range(data_dict.get("num_repeat_per_sample", 32) * 8):
            data = task_queue.get()
            if data == "STOP":
                break
            future = pool.schedule(
                _process_single_task, 
                args=(data, rollout_func, reward_func, remote_engine_url, sampling_params, data_dict)
            )
            active_futures.add(future)
        
        # 持续处理：任意任务完成就输出，并补充新任务
        while active_futures:
            # 检查所有活跃任务
            completed = {f for f in active_futures if f.done()}
            
            if completed:
                # 处理所有已完成的任务
                for future in completed:
                    try:
                        done_queue.put(future.result())
                    except Exception as e:
                        print(f"Error: {e}")
                    
                    active_futures.remove(future)
                
                # 补充新任务（补充与完成数量相同的任务）
                for _ in range(len(completed)):
                    data = task_queue.get()
                    if data == "STOP":
                        break
                    
                    future = pool.schedule(
                        _process_single_task, 
                        args=(data, rollout_func, reward_func, remote_engine_url, sampling_params, data_dict)
                    )
                    active_futures.add(future)
            else:
                # 没有任务完成，短暂休眠避免忙等待
                time.sleep(0.1)
    
    done_queue.put("COMPLETE")


class BaseGenerator:
    def __init__(
        self,
        remote_engine_url,
        remote_buffer_url,
        args: Optional[SimpleNamespace] = None,
        num_repeat_per_sample=1,
        queue_size=1024000,
        num_process=10,
        task_type="math",
        max_tokens=4096,
        num_epochs=10,
        skip_instance_ids: Optional[List[str]] = None,
    ):
        self.args = args
        self.queue_size = queue_size
        self.num_process = num_process
        self.remote_engine_url = remote_engine_url
        self.remote_buffer_url = remote_buffer_url
        self.num_repeat_per_sample = num_repeat_per_sample
        self.task_type = task_type
        self.max_tokens = max_tokens
        self.num_epochs = num_epochs
        
        # Ensure skip_instance_ids is a mutable list (copy to avoid modifying original)
        self.skip_instance_ids = list(skip_instance_ids) if skip_instance_ids is not None else None

        if self.skip_instance_ids is not None:
            print(f"BaseGenerator initialized with {len(self.skip_instance_ids)} instance_ids to skip")
            self.skip_instance_ids = self.skip_instance_ids * self.num_repeat_per_sample

        # init dataset using slime utils data
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.dataset = Dataset(
            args.prompt_data,
            tokenizer=self.tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs={},
            seed=args.rollout_seed,
        )
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )


    def send_data_to_buffer(self, data):
        remote_buffer_url = self.remote_buffer_url.rstrip("/") + "/buffer/write"

        for _ in range(3):
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

    def run(self, data_dict, rollout_func, reward_func):
        task_queue, done_queue = Queue(maxsize=self.queue_size), Queue(maxsize=self.queue_size)
        def read_data_into_queue():
            # read all data into queue
            for epoch_id in range(self.num_epochs):
                # shuffle data in each epoch
                self.dataset.shuffle(epoch_id)

                for sample in self.dataset:
                    # token in token out
                    sample.tokens = self.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
                    item = sample.to_dict()
                    item["instance_id"] = str(uuid.uuid4())

                    # put num_repeat_per_sample samples into queue
                    for _ in range(self.num_repeat_per_sample):
                        item_repeat = copy.deepcopy(item)
                        task_queue.put(item_repeat)

                time.sleep(600)

            for _ in range(self.num_process*self.num_process*self.num_repeat_per_sample):
                task_queue.put("STOP")

        processes = []
        for _ in range(self.num_process):
            process = Process(
                target=partial(worker_process, remote_engine_url=self.remote_engine_url, sampling_params=self.sampling_params.copy(), data_dict=data_dict),
                args=(task_queue, done_queue, rollout_func, reward_func),
            )
            process.start()
            processes.append(process)

        producer = Process(target=read_data_into_queue)
        producer.start()        

        # progress_bar = tqdm()
        num_finished = 0
        while num_finished < self.num_process:
            item = done_queue.get()
            if item == "COMPLETE":
                num_finished += 1
            else:
                assert "reward" in item, f"reward not in item: {item}"
                assert "instance_id" in item, f"instance_id not in item: {item}"
                self.send_data_to_buffer(item)
                # progress_bar.update(1)

        # progress_bar.close()
        return "finished"


    def entry(self, data, rollout_func, reward_func):
        _ = self.run(data, rollout_func, reward_func)


def run_rollout(data: dict):
    print(f"Starting math rollout with data: {data}")

    args_dict = data["args"]
    args = SimpleNamespace(**args_dict)

    generator = BaseGenerator(
        data["remote_engine_url"],
        data["remote_buffer_url"],
        args=args,
        queue_size=8192000,
        num_process=int(data.get("num_process", 100)),
        task_type=data["task_type"],
        num_epochs=int(data.get("num_epochs", 128)),
        num_repeat_per_sample=data.get("num_repeat_per_sample", 32),
        skip_instance_ids=data.get("skip_instance_ids", None),
    )

    generator.entry(data, math_rollout_func, get_rule_based_math_reward)


def is_valid_group(group, min_valid_group_size, task_type="math"):
    # Handle both tuple and list inputs
    if isinstance(group, tuple):
        instance_id, items = group
    else:
        items = group

    # Count valid items (non-empty responses)
    valid_indices = []
    for i, item in enumerate(items):
        if len(item["response"].strip()) > 0:
            valid_indices.append(i)

    group_size = len(items)
    valid_count = len(valid_indices)

    # A group is finished if it has reached the target size
    is_finished = group_size >= min_valid_group_size

    is_valid = is_finished and valid_count >= min_valid_group_size

    return is_valid
