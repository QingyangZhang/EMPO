"""Implement semantic entropy."""
import os
import pickle
import logging

import numpy as np
import wandb
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# We use the last GPU for semantic clustering
DEVICE = "cuda:{}".format(torch.cuda.device_count()-1) if torch.cuda.is_available() else "cpu"


class GeneralVerifier():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("TIGER-Lab/general-verifier")
        self.model = AutoModelForCausalLM.from_pretrained(
            "TIGER-Lab/general-verifier",
        ).to(DEVICE)
        self.model.eval()

    def check_implication(self, answer1, answer2, question, *args, **kwargs):

        # 创建带决策前缀的prompt（确保模型直接预测Yes/No）
        modified_prompt = (
            f"User: ### Question: {question}\n\n"
            f"### Ground Truth Answer: {answer1}\n\n"
            f"### Student Answer: {answer2}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            "If correct, output \"Final Decision: Yes\". If incorrect, output \"Final Decision: No\".\n"
            "Assistant: Final Decision: "  # 关键修改：将决策部分置于输入序列末尾
        )

        # 分词处理（注意添加return_offsets_mapping用于定位）
        inputs = self.tokenizer(modified_prompt, 
                      return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
    
        # 前向推理获取logits
        with torch.inference_mode():
            outputs = self.model.forward(input_ids)
        logits = outputs.logits  # [batch=1, seq_len=56, vocab_size=32000]
        
        # 确定预测位置（最后一个token的位置）
        predict_pos = input_ids.shape[1] - 1  # 对应"Final Decision: "后的预测位置
        
        # 提取目标位置的logits
        next_token_logits = logits[0, predict_pos, :]  # [vocab_size]
    
        # 获取Yes/No的token ID（考虑分词细节）
        decision_tokens = self.tokenizer(" Yes", " No", add_special_tokens=False) 
        yes_id = decision_tokens.input_ids[0]  # 假设"Yes"为单token
        no_id = decision_tokens.input_ids[1]   # 假设"No"为单token
    
        # 计算概率分布
        probs = torch.softmax(next_token_logits, dim=0)
        yes_prob = probs[yes_id].item()
        no_prob = probs[no_id].item()
        

        return yes_prob > no_prob


def are_equivalent(text1, text2, question, model):
    if text1 == '' or text2 == '':
        return False
    #implication_1 = model.check_implication(text1, text2, example=example)
    #implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
    #assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
    implication = model.check_implication(text1, text2, question)
    #if strict_entailment:
    #    semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
    #else:
    #    implications = [implication_1, implication_2]
    #    # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
    #    semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
    semantically_equivalent = implication

    return semantically_equivalent

def get_semantic_ids(strings_list, question, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j], question, model):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids

def get_semantic_ids_by_rule(strings_list, rule):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):
        semantically_equivalent = rule(text1, text2)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids

def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy