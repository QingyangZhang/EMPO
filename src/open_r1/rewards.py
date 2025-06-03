"""Reward functions for RL training."""

import math
import re
from typing import Dict
import numpy as np
import os

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from semantic_uncertainty import GeneralVerifier, get_semantic_ids, cluster_assignment_entropy, get_semantic_ids_by_rule, are_equivalent
from rouge import Rouge
rouge = Rouge()
import random

# Correctness reward for GRPO
def get_math_accuracy_reward(extract_answer=True):
    def accuracy_reward(completions, solution, **kwargs):
        """Accuracy reward function for mathematical reasoning tasks"""
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        predictions = []
        
        for content, sol in zip(contents, solution):
            # extract prediction
            result = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                    ],
                    extraction_mode="first_match",
                )

            if len(result) == 0:
                prediction = ''
            elif len(result) == 1:
                prediction = normalize_prediction(result[0])
            elif len(result) > 1:
                prediction = normalize_prediction(result[-1])
            predictions.append(prediction)
            
            # extract gold answer
            if extract_answer:
                gold_answer = parse(
                    sol,
                    extraction_mode="first_match",
                    extraction_config=[LatexExtractionConfig()],
                )
            else:
                gold_answer = solution
                
            if len(gold_answer) != 0:
                reward = float(verify(prediction, gold_answer))
            else:
                print('Fail to parse gold answer.')
                reward = 0.0
        
            rewards.append(reward)

        print("RANK: {}, Preds: {}, Golden Answers: {}, Reward: {}".format(local_rank, predictions, solution, rewards))

        return rewards

    return accuracy_reward

    
# Formatted Random Reward baseline suggested by "Spurious Rewards: Rethinking Training Signals in RLVR"
def get_random_math_reward(extract_answer=False):
    def random_math_reward(completions, solution, **kwargs):
        """Accuracy reward function for mathematical reasoning tasks"""
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        predictions = []
        
        for content, sol in zip(contents, solution):
            # extract prediction
            result = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                    ],
                    extraction_mode="first_match",
                )

            if len(result) == 0:
                prediction = ''
            elif len(result) == 1:
                prediction = normalize_prediction(result[0])
            elif len(result) > 1:
                prediction = normalize_prediction(result[-1])
            predictions.append(prediction)
            
            # extract gold answer
            if extract_answer:
                gold_answer = parse(
                    sol,
                    extraction_mode="first_match",
                    extraction_config=[LatexExtractionConfig()],
                )
            else:
                gold_answer = solution
                
            if len(gold_answer) != 0:
                if len(prediction) > 0:
                    reward = random.uniform(0,1)
                else:
                    reward = 0.0
            else:
                print('Fail to parse gold answer, skip.')
                reward = 0.0
        
            rewards.append(reward)

        print("RANK: {}, Preds: {}, Golden Answers: {}, Reward: {}".format(local_rank, predictions, solution, rewards))

        return rewards

    return random_math_reward

# Semantic Entropy Reward for our EMPO in Mathematical Reasoning
def get_empo_math_reward(num_generations):
    def semantic_entropy_math_reward(completions, problem, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        all_contents = [completion[0]["content"] for completion in completions]
        all_rewards = []

        for i in range(0,len(all_contents), num_generations):
            contents=all_contents[i:i+num_generations]

            rewards = []
            predictions = []
        
            for index, content in enumerate(contents):
                result = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                        ],
                        extraction_mode="first_match",
                    )

                if len(result) == 0:
                    prediction = ''
                elif len(result) == 1:
                    prediction = normalize_prediction(result[0])
                # if there are multiple answers in boxed, we extract the last one as final answer
                elif len(result) > 1:
                    prediction = normalize_prediction(result[-1])
                    
                predictions.append(prediction)
        
            semantic_ids = get_semantic_ids_by_rule(predictions, rule=verify)
            n_generations = len(semantic_ids)
            counts = np.bincount(semantic_ids)
            probabilities = counts/n_generations
            assert np.isclose(probabilities.sum(), 1)
            total_entropy = -(probabilities * np.log(probabilities)).sum()
        
            for index in range(len(contents)):
                # entropy thresholding to filter out highly unreliable answers
                if total_entropy < math.log(n_generations):
                    if predictions[index] == ''
                        reward = 0.0
                    else:
                        reward = probabilities[semantic_ids[index]]
                    
                    rewards.append(reward)
                else:
                    rewards.append(0.0)
        
            all_rewards.extend(rewards)
        print("RANK: {}, Contents: {}, Probability: {}, Semantic ID: {}, Reward: {}".format(local_rank, predictions, probabilities, semantic_ids, rewards))
        return all_rewards
        
    return semantic_entropy_math_reward

# This reward is only for visualization only    
def total_entropy_reward(completions, problem, **kwargs):
    """This function is used for visualization only."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    predictions = []
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    for index, content in enumerate(contents):
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        if len(answer_parsed) == 0:
            predictions.append('no answer {}'.format(index))
        else:
            predictions.append(answer_parsed)
    
    semantic_ids = get_semantic_ids_by_rule(predictions, rule=verify, strict_entailment=False)
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    total_entropy = -(probabilities * np.log(probabilities)).sum()
    
    for index in range(len(contents)):
        rewards.append(total_entropy)
    
    return rewards

# verifier is only used in natural reasoning tasks
#verifier = GeneralVerifier()

def normalize_prediction(final_answer):
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    return final_answer

# Reward for our EMPO in Natural Reasoning Tasks
def get_empo_common_reward(print_outputs=False):
    def semantic_prob_reward(completions, question, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        """Generalized reward function for free-form natural reasoning tasks."""
        if isinstance(completions[0], str):
            contents = [completion for completion in completions]
        else:
            contents = [completion[0]["content"] for completion in completions]
        predictions = []
        lengths = []
        # extract content in box
        for index, content in enumerate(contents):
            result = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=False,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            if len(result) == 0:
                prediction = ''
            elif len(result) == 1:
                prediction = normalize_prediction(result[0])
            elif len(result) > 1:
                prediction = normalize_prediction(result[-1])
            predictions.append(prediction)
            lengths.append(len(verifier.tokenizer.tokenize(prediction)))
            
        rewards = []
        try:
            semantic_ids = get_semantic_ids(predictions, question[0], verifier, strict_entailment=False)
        except:
            print('Fail to cluster.')
            return [0.0] * len(predictions)
        n_generations = len(semantic_ids)
        counts = np.bincount(semantic_ids)
        probabilities = counts/n_generations
        assert np.isclose(probabilities.sum(), 1)
        max_prob = np.max(probabilities)
        max_prob_indices = np.where(probabilities == max_prob)[0]
        total_entropy = -(probabilities * np.log(probabilities)).sum()

        normalized_lengths = [(x - min(lengths)) / (max(lengths) - min(lengths) + 1e-10) + 1 for x in lengths]
        for index in range(len(contents)):
            try:
                rouge_score = rouge.get_scores(predictions[index].lower(), question[0].lower())
                rep = rouge_score[0]["rouge-l"]["p"]
            except:
                rep = 1.0
            if predictions[index] == '':
                rewards.append(-0.5)
            elif rep > 0.8 or predictions[index].lower() in question[0].lower():
                rewards.append(0.0)
            elif max_prob < 0.2 and len(probabilities) > 1:
                rewards.append(0.0)
            else:
                reward = probabilities[semantic_ids[index]] #* normalized_lengths[index]
                rewards.append(reward)

        if print_outputs:
            print("RANK: {},\n Question: {},\n Output: {}, Answers: {},\n Probability: {},\n Semantic ID: {}, \n Reward: {}\n\n".
                  format(local_rank, question[0], contents[0], predictions, probabilities, semantic_ids, rewards))
    
        return rewards

    return semantic_prob_reward

# Formatted Random Reward suggested by "Spurious Rewards: Rethinking Training Signals in RLVR"
def get_random_general_reward():
    def random_reward(completions, reference_answer, question, **kwargs):
        """Generalized accuracy reward function for free-form natural reasoning tasks."""
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        predictions = []
        gold_answers = reference_answer
        
        for content, sol in zip(contents, reference_answer):
            # extract prediction
            result = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=False,
                        equations=True,
                        boxed="all",
                        units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                    ],
                    extraction_mode="first_match",
                )
            
            if len(result) == 0:
                prediction = ''
            elif len(result) == 1:
                prediction = normalize_prediction(result[0])
            elif len(result) > 1:
                prediction = normalize_prediction(result[-1])
            predictions.append(prediction)

        for prediction, gold_answer in zip(predictions, gold_answers):
            try:
                if prediction == '':
                    reward = -0.5
                else:
                    reward = random.uniform(0,1)
            except:
                print(f'RANK {local_rank}: Skip over-long answer to avoid OOM.')
                return [0.0] * len(predictions)
        
            rewards.append(reward)

        print("RANK: {},\n Question: {},\n Output: {}, Answers: {},\n Reward: {}\n\n".
                  format(local_rank, question[0], predictions, gold_answers, rewards))

        return rewards

    return random_reward

# Correctness Reward for GRPO
def get_general_accuracy_reward():
    def accuracy_reward(completions, reference_answer, question, **kwargs):
        """Generalized accuracy reward function for free-form natural reasoning tasks."""
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        predictions = []
        gold_answers = reference_answer
        
        for content, sol in zip(contents, reference_answer):
            # extract prediction
            result = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=False,
                        equations=True,
                        boxed="all",
                        units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                    ],
                    extraction_mode="first_match",
                )
            
            if len(result) == 0:
                prediction = ''
            elif len(result) == 1:
                prediction = normalize_prediction(result[0])
            elif len(result) > 1:
                prediction = normalize_prediction(result[-1])
            predictions.append(prediction)

        for prediction, gold_answer in zip(predictions, gold_answers):
            try:
                if are_equivalent(prediction, gold_answer, question[0], verifier):
                    reward = 1.0
                elif prediction == '':
                    reward = -0.5
                else:
                    reward = 0.0
            except:
                print(f'RANK {local_rank}: Skip over-long answer to avoid OOM.')
                return [0.0] * len(predictions)
        
            rewards.append(reward)

        print("RANK: {},\n Question: {},\n Output: {}, Answers: {},\n Reward: {}\n\n".
                  format(local_rank, question[0], predictions, gold_answers, rewards))

        return rewards

    return accuracy_reward