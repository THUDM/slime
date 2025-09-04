from typing import List
import time
import re
import asyncio
from .prompts import SOLVER_PROMPT_TEMPLATE, generate_rewriter_template, generate_select_template
from copy import deepcopy
import traceback

from slime.utils.http_utils import post
from slime.utils.types import Sample
from slime.rollout.rm_hub import batched_async_rm


async def generate_response(args, prompt, key):
    try:
        sampling_params = args.sampling_params
        tokenizer = args.tokenizer
        max_context_length = args.rollout_max_context_len
        sample = deepcopy(args.sample)

        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        sample.tokens = prompt_token_ids
        sample.prompt = prompt
        input_token_ids = prompt_token_ids
        prompt_length = len(input_token_ids)
        current_sampling_params = deepcopy(sampling_params)
        current_sampling_params["max_new_tokens"] = min(
            sampling_params["max_new_tokens"], max_context_length - prompt_length
        )

        if current_sampling_params["max_new_tokens"] <= 0:
            return None

        payload = {"input_ids": input_token_ids, "sampling_params": current_sampling_params, "return_logprob": True}

        output = await post(url, payload, use_http2=args.use_http2)

        # Extract new response tokens
        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            # abort
            new_response_tokens = []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response = output["text"]

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            # case "abort":
            #     sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

        args.results_dict[key].append(sample)

        final = output["text"].replace("<|user|>", "")
        if "</think>" in final:
            contents = final.split("</think>")
            if len(contents) == 2 and contents[1] != "":
                reason_content = contents[0].strip()
                response_content = contents[1].strip()
                sample.reason_content = reason_content
                sample.response_content = response_content
                return response_content
        sample.reason_content = None
        sample.response_content = None
        return None
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


class Agent:
    """A base class for our AI agents."""

    def __init__(self):
        pass

    async def run(self, args, prompt, max_retries: int = 1, key: str = None) -> str:
        """Runs the agent by sending a prompt to the LLM."""
        for i in range(max_retries):
            try:
                response = await generate_response(args, prompt, key=key)
                return response
            except Exception as e:
                print(f"Error querying LLM: {e}")
                time.sleep(1)
        print(f"Failed to query LLM after {max_retries} retries")
        return None


class SolverAgent(Agent):
    """The agent responsible for generating and improving solutions."""

    def __init__(self):
        super().__init__()

    async def generate_initial_solution(self, args, problem_statement) -> str:
        """Generates the first solution attempt."""
        prompt = SOLVER_PROMPT_TEMPLATE.format(problem_statement=problem_statement)
        return await self.run(args, prompt, max_retries=3, key="solver")


class RewriterAgent(Agent):
    """The agent responsible for rewriting solutions."""

    def __init__(self):
        super().__init__()

    async def rewrite(self, args, problem_statement, previous_solutions: List[str]) -> str:
        """Generates the rewrited solution."""

        # 动态生成模板
        template = generate_rewriter_template(len(previous_solutions))

        # 构建参数字典
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(previous_solutions):
            format_params[f"solution{i+1}"] = solution

        prompt = template.format(**format_params)
        return await self.run(args, prompt, max_retries=1, key="rewriter")


class SelectorAgent(Agent):
    """The agent responsible for selecting solutions."""

    def __init__(self):
        super().__init__()

    async def select(self, args, problem_statement, candidate_solutions: List[str]) -> str:
        """Generates the rewrited solution."""

        # 动态生成模板
        template = generate_select_template(len(candidate_solutions))

        # 构建参数字典
        format_params = {"problem_statement": problem_statement}
        for i, solution in enumerate(candidate_solutions):
            format_params[f"solution{i+1}"] = solution

        prompt = template.format(**format_params)
        return await self.run(args, prompt, max_retries=10, key="selector")

    def extract_selected_solution_idx(self, response: str, candidate_solutions: List[str]) -> int:
        """Extracts the selected solution ID from the response."""
        PATTERN = re.compile("Judgment:\s*(\d+)")
        matched = PATTERN.findall(response)
        try:
            selected_id = int(matched[0]) - 1
            if selected_id < len(candidate_solutions) and selected_id >= 0:
                return selected_id
            else:
                return None
        except Exception as e:
            print(f"extract_selected_solution_idx error: {e}")
            return None


async def rewrite_worker(args, previous_solutions, problem_statement, worker_id):
    rewriter = RewriterAgent()
    new_solution = await rewriter.rewrite(args, problem_statement, previous_solutions)
    return new_solution


async def solver_worker(args, problem_statement, worker_id):
    """
    单组 solver 流程。
    """

    try:
        solver = SolverAgent()
        current_solution = await solver.generate_initial_solution(args, problem_statement)
        return current_solution

    except Exception as e:
        print(f"[Worker-{worker_id}] exception: {e}")
        print(f"[Worker-{worker_id}] traceback: {traceback.format_exc()}")
        return None


async def run_agent_system(args, sample):
    """
    并发运行 num_parallel 组 pipeline。
    """

    args = deepcopy(args)  # 深拷贝 args，因为 args 在 rollout_with_multi_agents 中会被修改
    args.sample = sample
    args.results_dict = {"solver": [], "rewriter": [], "selector": []}

    problem_statement = sample.prompt
    tasks = [solver_worker(args, problem_statement, worker_id) for worker_id in range(args.num_parallel)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for sample, reward in zip(args.results_dict["solver"], rewards):
        sample.reward = reward

    previous_solutions = [item for item in results if isinstance(item, str)]

    def reward_adjustment(samples, reward_weight):
        for sample in samples:
            sample.reward = sample.reward * reward_weight
        return samples

    if len(previous_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        return args.results_dict["solver"]

    # Rewriting
    tasks = [
        rewrite_worker(args, previous_solutions, problem_statement, worker_id)
        for worker_id in range(args.num_parallel)
    ]
    rewrited_solutions_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常结果
    rewrited_solutions = []
    for i, result in enumerate(rewrited_solutions_raw):
        if isinstance(result, str):
            rewrited_solutions.append(result)

    rewards = await batched_async_rm(args, args.results_dict["rewriter"])
    for sample, reward in zip(args.results_dict["rewriter"], rewards):
        sample.reward = reward

    if len(rewrited_solutions) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        return args.results_dict["solver"] + args.results_dict["rewriter"]

    # Selection
    selector = SelectorAgent()
    response = await selector.select(args, problem_statement, rewrited_solutions)
    if len(args.results_dict["selector"]) == 0:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        return args.results_dict["solver"] + args.results_dict["rewriter"]

    assert (
        len(args.results_dict["selector"]) == 1
    ), f"selector should only return one solution, but got {len(args.results_dict['selector'])}"
    if response is None:
        args.results_dict["selector"][0].reward = 0
    else:
        selected_solution_idx = selector.extract_selected_solution_idx(response, rewrited_solutions)
        if selected_solution_idx is None:
            args.results_dict["selector"][0].reward = 0
        else:
            selected_solution = rewrited_solutions[selected_solution_idx]
            for sample in args.results_dict["rewriter"]:
                if sample.response_content is not None and selected_solution in sample.response_content:
                    args.results_dict["selector"][0].reward = sample.reward
                    break

    ## 如果最终答案正确，对所有reward添加正向的奖励；如果最终答案不正确，对所有reward添加负向的惩罚。
    if args.results_dict["selector"][0].reward == 1:
        reward_adjustment(args.results_dict["solver"], args.correct_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.correct_reward_weight)
        reward_adjustment(args.results_dict["selector"], args.correct_reward_weight)
    else:
        reward_adjustment(args.results_dict["solver"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["rewriter"], args.incorrect_reward_weight)
        reward_adjustment(args.results_dict["selector"], args.incorrect_reward_weight)

    return args.results_dict["solver"] + args.results_dict["rewriter"] + args.results_dict["selector"]
