"""
KIE (Key Information Extraction) Reward Functions

针对 OCR/文档理解任务的奖励函数，支持 JSON 格式的键值对提取评估。
"""
import json
import re
import logging
from typing import Any
from difflib import SequenceMatcher

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """标准化文本：去除空白、统一大小写等"""
    if text is None:
        return ""
    text = str(text).strip()
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_value(value: Any) -> str:
    """标准化值，处理各种类型"""
    if value is None:
        return ""
    if isinstance(value, list):
        # 列表类型，递归处理
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return normalize_text(str(value))


def extract_json_from_response(response: str) -> dict | None:
    """从模型响应中提取 JSON"""
    if not response:
        return None

    # 尝试直接解析
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'\{[\s\S]*\}',                   # { ... }
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    return None


def string_similarity(s1: str, s2: str) -> float:
    """计算两个字符串的相似度 (0-1)"""
    s1 = normalize_text(s1).lower()
    s2 = normalize_text(s2).lower()

    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    return SequenceMatcher(None, s1, s2).ratio()


def compute_value_score(pred_value: Any, gt_value: Any, strict: bool = False) -> float:
    """
    计算单个值的匹配分数

    Args:
        pred_value: 预测值
        gt_value: 真实值
        strict: 是否严格匹配

    Returns:
        float: 0-1 之间的分数
    """
    pred_str = normalize_value(pred_value)
    gt_str = normalize_value(gt_value)

    # 空值处理
    if not gt_str:
        # 真实值为空，预测也为空则正确
        return 1.0 if not pred_str else 0.5

    if not pred_str:
        # 真实值非空但预测为空
        return 0.0

    # 精确匹配
    if pred_str.lower() == gt_str.lower():
        return 1.0

    # 严格模式下不给部分分
    if strict:
        return 0.0

    # 包含关系
    if gt_str.lower() in pred_str.lower() or pred_str.lower() in gt_str.lower():
        return 0.8

    # 字符串相似度
    similarity = string_similarity(pred_str, gt_str)
    if similarity > 0.8:
        return similarity

    return 0.0


def compute_list_score(pred_list: list, gt_list: list) -> float:
    """计算列表类型值的匹配分数"""
    if not gt_list:
        return 1.0 if not pred_list else 0.5

    if not pred_list:
        return 0.0

    # 对于列表，计算最佳匹配
    total_score = 0.0
    matched_pred = set()

    for gt_item in gt_list:
        best_score = 0.0
        best_idx = -1

        for idx, pred_item in enumerate(pred_list):
            if idx in matched_pred:
                continue

            if isinstance(gt_item, dict) and isinstance(pred_item, dict):
                score = compute_dict_score(pred_item, gt_item)
            else:
                score = compute_value_score(pred_item, gt_item)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0:
            matched_pred.add(best_idx)
        total_score += best_score

    # 惩罚多余的预测
    extra_penalty = max(0, len(pred_list) - len(gt_list)) * 0.1

    return max(0, total_score / len(gt_list) - extra_penalty)


def compute_dict_score(pred_dict: dict, gt_dict: dict, weights: dict = None) -> float:
    """
    计算字典（JSON对象）的匹配分数

    Args:
        pred_dict: 预测的字典
        gt_dict: 真实的字典
        weights: 各个 key 的权重，默认均等

    Returns:
        float: 0-1 之间的分数
    """
    if not gt_dict:
        return 1.0 if not pred_dict else 0.5

    if not pred_dict:
        return 0.0

    total_score = 0.0
    total_weight = 0.0

    for key, gt_value in gt_dict.items():
        weight = weights.get(key, 1.0) if weights else 1.0
        total_weight += weight

        if key not in pred_dict:
            # 缺失 key
            continue

        pred_value = pred_dict[key]

        # 根据值类型计算分数
        if isinstance(gt_value, list):
            pred_list = pred_value if isinstance(pred_value, list) else [pred_value]
            score = compute_list_score(pred_list, gt_value)
        elif isinstance(gt_value, dict):
            pred_d = pred_value if isinstance(pred_value, dict) else {}
            score = compute_dict_score(pred_d, gt_value)
        else:
            score = compute_value_score(pred_value, gt_value)

        total_score += score * weight

    # 惩罚多余的 key
    extra_keys = set(pred_dict.keys()) - set(gt_dict.keys())
    extra_penalty = len(extra_keys) * 0.05

    final_score = (total_score / total_weight) - extra_penalty if total_weight > 0 else 0.0
    return max(0.0, min(1.0, final_score))


# ============== 主要的奖励函数 ==============

async def kie_reward(args, sample: Sample, **kwargs) -> float:
    """
    KIE 任务的奖励函数

    评估维度：
    1. JSON 格式正确性
    2. Key 完整性
    3. Value 准确性

    支持的 GT 格式：
    1. JSON Dict: {"key": "value"}
    2. JSON 嵌套 List: {"items": [{"name": "...", "price": "..."}]}
    3. 纯文本: "张三"

    Args:
        args: 训练参数
        sample: Sample 对象
            - sample.response: 模型生成的响应
            - sample.label: 正确答案（JSON 字符串或纯文本）

    Returns:
        float: 0-1 之间的奖励值
    """
    response = sample.response
    label = sample.label

    # 解析真实答案
    gt_dict = None
    gt_is_plain_text = False

    if isinstance(label, str):
        try:
            gt_dict = json.loads(label)
        except json.JSONDecodeError:
            # 纯文本 GT，使用字符串相似度评估
            gt_is_plain_text = True
    elif isinstance(label, dict):
        gt_dict = label
    else:
        logger.warning(f"Unexpected label type: {type(label)}")
        return 0.0

    # 处理纯文本 GT
    if gt_is_plain_text:
        gt_text = normalize_text(label)
        pred_text = normalize_text(response)

        if not gt_text:
            return 1.0 if not pred_text else 0.5

        # 精确匹配
        if pred_text.lower() == gt_text.lower():
            return 1.0

        # 包含关系
        if gt_text.lower() in pred_text.lower():
            return 0.9

        # 字符串相似度
        similarity = string_similarity(pred_text, gt_text)
        return similarity

    # 解析模型响应
    pred_dict = extract_json_from_response(response)

    # JSON 格式奖励
    if pred_dict is None:
        # 无法解析 JSON，给少量分数（如果响应中包含部分正确内容）
        format_reward = 0.0
        # 检查是否包含任何正确的值
        for key, value in gt_dict.items():
            if str(value) and str(value) in response:
                format_reward = 0.1
                break
        return format_reward

    # 计算内容分数
    content_score = compute_dict_score(pred_dict, gt_dict)

    # 格式正确性加分
    format_bonus = 0.1  # JSON 格式正确的基础分

    final_reward = min(1.0, content_score * 0.9 + format_bonus)

    return final_reward


async def kie_reward_strict(args, sample: Sample, **kwargs) -> float:
    """
    严格版本的 KIE 奖励函数

    只有完全匹配才给分，适合简单任务或后期训练
    支持纯文本 GT
    """
    response = sample.response
    label = sample.label

    # 解析
    gt_dict = None
    gt_is_plain_text = False

    if isinstance(label, str):
        try:
            gt_dict = json.loads(label)
        except json.JSONDecodeError:
            gt_is_plain_text = True
    else:
        gt_dict = label

    # 处理纯文本 GT
    if gt_is_plain_text:
        gt_text = normalize_text(label)
        pred_text = normalize_text(response)
        return 1.0 if pred_text.lower() == gt_text.lower() else 0.0

    pred_dict = extract_json_from_response(response)

    if pred_dict is None:
        return 0.0

    # 严格比较
    correct_keys = 0
    total_keys = len(gt_dict)

    for key, gt_value in gt_dict.items():
        if key in pred_dict:
            pred_value = pred_dict[key]
            if normalize_value(pred_value).lower() == normalize_value(gt_value).lower():
                correct_keys += 1

    return correct_keys / total_keys if total_keys > 0 else 0.0


async def kie_reward_f1(args, sample: Sample, **kwargs) -> float:
    """
    基于 F1 分数的 KIE 奖励函数

    计算 key-value 对的 Precision, Recall, F1
    支持纯文本 GT
    """
    response = sample.response
    label = sample.label

    # 解析
    gt_dict = None
    gt_is_plain_text = False

    if isinstance(label, str):
        try:
            gt_dict = json.loads(label)
        except json.JSONDecodeError:
            gt_is_plain_text = True
    else:
        gt_dict = label

    # 处理纯文本 GT - 使用字符串相似度作为 F1
    if gt_is_plain_text:
        gt_text = normalize_text(label)
        pred_text = normalize_text(response)

        if not gt_text:
            return 1.0 if not pred_text else 0.0

        if pred_text.lower() == gt_text.lower():
            return 1.0

        return string_similarity(pred_text, gt_text)

    pred_dict = extract_json_from_response(response)

    if pred_dict is None:
        return 0.0

    # 计算 TP, FP, FN
    tp = 0  # 正确预测的 key-value 对
    fp = 0  # 错误预测的
    fn = 0  # 漏掉的

    for key, gt_value in gt_dict.items():
        gt_str = normalize_value(gt_value)
        if not gt_str:  # 跳过空值
            continue

        if key in pred_dict:
            pred_str = normalize_value(pred_dict[key])
            if pred_str.lower() == gt_str.lower():
                tp += 1
            else:
                # 部分匹配
                sim = string_similarity(pred_str, gt_str)
                if sim > 0.8:
                    tp += sim
                    fp += (1 - sim)
                else:
                    fp += 1
                    fn += 1
        else:
            fn += 1

    # 多余的预测
    for key in pred_dict:
        if key not in gt_dict:
            pred_str = normalize_value(pred_dict[key])
            if pred_str:  # 非空预测
                fp += 0.5  # 轻微惩罚

    # 计算 F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


async def kie_reward_weighted(args, sample: Sample, **kwargs) -> float:
    """
    加权版本的 KIE 奖励函数

    可以通过 metadata 指定不同 key 的权重
    """
    response = sample.response
    label = sample.label
    metadata = sample.metadata or {}

    # 从 metadata 获取权重配置
    key_weights = metadata.get("key_weights", {})

    # 解析
    if isinstance(label, str):
        try:
            gt_dict = json.loads(label)
        except json.JSONDecodeError:
            return 0.0
    else:
        gt_dict = label

    pred_dict = extract_json_from_response(response)

    if pred_dict is None:
        return 0.0

    return compute_dict_score(pred_dict, gt_dict, weights=key_weights)


# ============== 批量版本 ==============

async def kie_reward_batch(args, samples: list[Sample], **kwargs) -> list[float]:
    """批量计算 KIE 奖励"""
    rewards = []
    for sample in samples:
        reward = await kie_reward(args, sample, **kwargs)
        rewards.append(reward)
    return rewards
