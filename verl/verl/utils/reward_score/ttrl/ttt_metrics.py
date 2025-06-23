from collections import Counter
from typing import List
import math

from verl.utils.reward_score.ttrl.auto_extract import auto_extract
from verl.utils.reward_score.ttrl.auto_verify import auto_verify    

def semantic_cluster(task, model_answers, extra_info):
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
            if auto_verify(task, [ans], [rep], extra_info=extra_info)[0][0]:
                counts[idx] += 1          # 增加计数
                cluster_indices.append(idx)  # 记录聚类索引
                found = True
                break
        
        # 未找到匹配则创建新聚类
        if not found:
            representatives.append(ans)  # 添加新代表
            counts.append(1)              # 新聚类计数为1
            cluster_indices.append(len(representatives) - 1)  # 记录新索引

    # 计算每个答案的频率（长度为n的列表）
    frequencies = [counts[idx] / n for idx in cluster_indices]
    
    # 计算每个独特答案的频率（长度为len(representatives)的列表）
    unique_frequencies = [c / n for c in counts]
    
    # 返回：每个答案的频率列表，所有独特答案列表，独特答案频率列表
    return frequencies, representatives, unique_frequencies

def entropy_thresholding(frequencies, unique_frequencies, low, high):
    n = len(unique_frequencies)

    entropy = 0.0
    for p in unique_frequencies:
        if p > 0:
            entropy -= p * math.log(p)
    
    max_entropy = math.log(n)
    
    min_valid = low * max_entropy
    max_valid = high * max_entropy
    
    if entropy >= min_valid and entropy <= max_valid:
        return frequencies
    else:
        return [0.0] * len(frequencies)


def test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    task="math", extra_info=None, reward_type=None, entropy_thres=None):
    
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"

    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)
    counter = Counter(model_answers)
    
    estimated_label, majority_count = counter.most_common(1)[0]
    
    hit_rate = 1.0 if auto_verify(task, [estimated_label], [ground_truth], extra_info=extra_info)[0][0] else 0.0
    majority_ratio = majority_count / len(solutions)
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)
    
    assert reward_type in ['gt', 'semantic entropy', 'voting']
    if reward_type == 'voting':
        # TTRL rewards
        rewards, _ = auto_verify(task, solutions, [estimated_label] * len(solutions), extra_info=extra_info)
    elif reward_type == 'semantic entropy':
        # EMPO rewards
        frequencies, unique_answers, unique_frequencies = semantic_cluster(task, model_answers, extra_info)
        rewards = entropy_thresholding(frequencies, unique_frequencies, low=0.0, high=entropy_thres)
    elif reward_type == 'gt':
        # true rewards
        rewards = true_rewards

    
    rewards_hit_rate = 0
    for reward, true_reward in zip(rewards, true_rewards):
        if reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(rewards)

    assert len(rewards) == len(solutions), f"{len(rewards)} vs {len(solutions)}"

    ttrl_metrics = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_ratio": majority_ratio,
        "mean_train_accuracy": sum(true_rewards) / len(true_rewards),
        "mean_reward": sum(rewards) / len(rewards),
        f"pass@{len(solutions)}": 1.0 if sum(true_rewards) >= 1 else 0.0,
    }
    return rewards, ttrl_metrics

def post_test_time_train_metrics(
    solutions: List[str],
    ground_truth: List[str],
    pred_rewards: List,
    task="math", extra_info=None):
    assert len(solutions) == len(ground_truth), f"{len(solutions)} vs {len(ground_truth)}"
    assert len(solutions) == len(pred_rewards), f"{len(solutions)} vs {len(pred_rewards)}"
    assert len(set(ground_truth)) == 1, f"Ground truth is not unique: {ground_truth}"
    ground_truth = ground_truth[0]

    model_answers = auto_extract(task, solutions, extra_info=extra_info)

    counter = Counter(model_answers)
    
    # true_label_ratio = counter.get(ground_truth, 0) / len(solutions)

    true_rewards, _ = auto_verify(task, solutions, [ground_truth] * len(solutions), extra_info=extra_info)

    # Compare pred_rewards with true_rewards to calculate reward hit rate
    rewards_hit_rate = sum(
        1 if pred == true else 0 for pred, true in zip(pred_rewards, true_rewards)
    ) / len(pred_rewards)

    post_ttrl_metrics = {
        "post_reward_accuracy": rewards_hit_rate,
        "post_mean_train_accuracy": sum(true_rewards) / len(true_rewards),
        f"post_pass@{len(solutions)}": 1.0 if sum(true_rewards) > 0 else 0.0,
    }
    return post_ttrl_metrics