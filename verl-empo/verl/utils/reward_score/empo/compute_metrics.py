from typing import List
import math
import numpy as np
import random

from verl.utils.reward_score.empo.auto_extract import auto_extract
from verl.utils.reward_score.empo.auto_verify import auto_verify

import traceback

def binarize_max(frequencies):
    arr = np.array(frequencies)
    max_val = np.max(arr)  # 获取最大值[7,9](@ref)
    return np.where(arr == max_val, 1, 0).tolist()  # 向量化替换

def semantic_cluster(model_answers, extra_info):
    representatives = []  # 存储独特的答案代表（包括每个空字符串）
    counts = []          # 存储每个独特答案对应的出现次数
    cluster_indices = []  # 存储每个答案对应的聚类索引

    n = len(model_answers)  # 总答案数量
    
    for i, ans in enumerate(model_answers):
        # 处理空字符串：每个空字符串都视为独特聚类
        if ans == "":
            representatives.append(ans)  # 添加空字符串作为新代表
            counts.append(1)              # 出现次数为1
            cluster_indices.append(len(representatives) - 1)  # 记录新聚类索引
            continue
        
        # 处理非空字符串：尝试匹配已有聚类
        found = False
        for idx, rep in enumerate(representatives):
            # 跳过空字符串代表（避免非空字符串与空字符串比较）
            if rep == "":
                continue
            # 使用auto_verify判断答案是否匹配
            rs, _ = auto_verify([ans], [rep], extra_info=extra_info)
            if rs[0]:
                counts[idx] += 1          # 增加计数
                cluster_indices.append(idx)  # 记录聚类索引
                found = True
                #print(f"{[ans]} equal to {[rep]}")
                break
            else:
                #print(f"{[ans]} not equal to {[rep]}")
                pass
        
        # 未找到匹配则创建新聚类
        if not found:
            representatives.append(ans)  # 添加新代表
            counts.append(1)              # 新聚类计数为1
            cluster_indices.append(len(representatives) - 1)  # 记录新索引

    # 计算每个答案的频率（长度为n的列表）
    frequencies = [counts[idx] / n for idx in cluster_indices]
    
    # 计算每个独特答案的频率（长度为len(representatives)的列表）
    unique_frequencies = [c / n for c in counts]
    assert len(representatives) == len(unique_frequencies)
    # print(f"model answers: {model_answers}")
    # print(f"frequencies: {frequencies}")
    # 返回：每个答案的频率列表，所有独特答案列表，独特答案频率列表
    return frequencies, representatives, unique_frequencies

def entropy_thresholding(frequencies, unique_frequencies):
    n = len(unique_frequencies)

    entropy = 0.0
    for p in unique_frequencies:
        if p > 0:
            entropy -= p * math.log(p)
    
    max_entropy = math.log(n)
    
    # min_valid = low * max_entropy
    # max_valid = high * max_entropy
    confidence = max(frequencies)
    
    if confidence >= 0.25:
        return frequencies, 0
    else:
        return [0.0] * len(frequencies), 1

def post_semantic_cluster(model_answers, equality):
    # 存储每个答案的聚类索引（初始为-1）
    cluster_indices = [-1] * len(model_answers)
    # 存储每个聚类的频率（长度等于聚类数量）
    cluster_counts = []
    # 存储每个聚类的代表（第一个加入该聚类的答案）
    representatives = []
    
    n = len(model_answers)
    
    # 遍历每个答案
    for i, ans in enumerate(model_answers):
        if ans == "":
            # 对于空字符串，直接创建新聚类
            cluster_indices[i] = len(cluster_counts)
            cluster_counts.append(1)
            representatives.append(ans)
            continue
            
        # 尝试加入已有聚类（只检查非空聚类）
        found = False
        for j in range(len(cluster_counts)):
            if representatives[j] == "":
                continue  # 跳过空聚类
            rep_index = cluster_indices.index(j)  # 找到该聚类的第一个成员索引
            if equality[i][rep_index]:
                cluster_indices[i] = j
                cluster_counts[j] += 1
                found = True
                break
                
        # 未找到匹配则创建新聚类
        if not found:
            cluster_indices[i] = len(cluster_counts)
            cluster_counts.append(1)
            representatives.append(ans)
    
    # 计算每个答案的频率（长度等于答案数量）
    frequencies = [cluster_counts[idx] / n for idx in cluster_indices]
    # 计算每个聚类的频率（长度等于聚类数量）
    unique_frequencies = [count / n for count in cluster_counts]
    
    return frequencies, representatives, unique_frequencies

def run_sequential_auto_verify(
    auto_verify_func,
    model_answers: list[str],
    extra_info=None,
    # timeout and num_workers are not used but kept for signature compatibility
    timeout: int = 60,
    num_workers: int = 1
) -> list[list[int]]:
    """
    Performs sequential (single-process) auto-verification. Useful for debugging.
    """
    N = len(model_answers)
    result_matrix = [[0] * N for _ in range(N)]

    # Use tqdm for a progress bar, as this can be slow
    # pbar = tqdm(total=(N * (N - 1) // 2), desc="Sequential Auto-Verification")
    
    for i in range(N):
        result_matrix[i][i] = 1 # An answer is always equivalent to itself
        for j in range(i + 1, N):
            try:
                # Directly call the function
                result = auto_verify_func([model_answers[i]], [model_answers[j]], extra_info)
                
                # Handle both tuple and int return types
                value = result[0][0] if isinstance(result, tuple) else result
                result_matrix[i][j] = value
                result_matrix[j][i] = value # Similarity Matrix is symmetric

            except Exception:
                e = traceback.format_exc()
                print(f"\n[Error] Verification for pair ({i}, {j}) failed: {e}")
                result_matrix[i][j] = 0
                result_matrix[j][i] = 0
                
    return result_matrix


def empo_metrics(
    solutions: List[str],
    ground_truth: List[str],
    extra_info=None, reward_type=None, entropy_thres=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    if isinstance(ground_truth[0], list):
        ground_truth = [gt[-1] for gt in ground_truth]
    assert (
        isinstance(ground_truth, list) and
        all(isinstance(item, str) for item in ground_truth)
    ), "Ground truth must be list[str]"
    
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {set(ground_truth)}"
    ground_truth_str = ground_truth[0]
    
    model_answers = auto_extract(solutions, extra_info=extra_info)
    assert (
        isinstance(model_answers, list) and
        all(isinstance(item, str) for item in model_answers)
    ), "Model answers must be list[str]"

    answers_with_gt = model_answers + [ground_truth_str]
    result_matrix = run_sequential_auto_verify(auto_verify, answers_with_gt, extra_info=extra_info)
    equality_matrix = [row[:-1] for row in result_matrix[:-1]]
    true_rewards = result_matrix[-1][:-1]
    assert len(true_rewards) == len(model_answers)

    frequencies, unique_answers, unique_frequencies = post_semantic_cluster(model_answers, equality_matrix)
    
    # Handle case where frequencies is empty
    if not frequencies:
        hit_rate = 0.0
        majority_ratio = 0.0
        estimated_label = ""
    else:
        max_index = np.argmax(frequencies)
        estimated_label = model_answers[max_index]
        majority_ratio = frequencies[max_index]
        hit_rate = float(true_rewards[max_index])

    assert reward_type in ['gt', 'entropy', 'voting', 'format', 'random', 'best']
    if reward_type == 'voting':
        rewards = binarize_max(frequencies)
    elif reward_type == 'entropy':
        # rewards, filtered = entropy_thresholding(frequencies, unique_frequencies)
        rewards = frequencies #, unique_frequencies
    elif reward_type == 'gt':
        rewards = true_rewards
    elif reward_type == 'best':
        rewards = true_rewards if hit_rate < 1 else [0.0] * len(true_rewards)
    elif reward_type == 'random':
        rewards = [random.uniform(0, 1) for _ in range(len(true_rewards))]
    elif reward_type == 'format':
        rewards = [1.0 if len(ans) > 0 else 0 for ans in model_answers]

    rewards_hit_rate = sum(1 for r, tr in zip(rewards, true_rewards) if r == tr) / len(rewards) if rewards else 0

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    metrics = {
        "label_accuracy": [hit_rate] * len(model_answers),
        "reward_accuracy": [rewards_hit_rate] * len(model_answers),
        "majority_ratio": [majority_ratio] * len(model_answers),
        "train_accuracy": [sum(true_rewards) / len(true_rewards) if true_rewards else 0.0] * len(model_answers),
        "train_reward": [sum(rewards) / len(rewards) if rewards else 0.0] * len(model_answers),
        f"pass@{len(solutions)}": [1.0 if sum(true_rewards) >= 1 else 0.0] * len(model_answers),
        "extracted_answers": model_answers,
        "estimated_label": [estimated_label] * len(model_answers),
    }

    return rewards, metrics