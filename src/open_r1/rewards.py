"""Reward functions for GRPO training."""

import math
import re
from typing import Dict
import numpy as np
import os

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from semantic_uncertainty import EntailmentDeberta, GeneralVerifier, get_semantic_ids, cluster_assignment_entropy, get_semantic_ids_by_rule, are_equivalent
from rouge import Rouge
rouge = Rouge()


def get_math_accuracy_reward(extract_answer=True):
    def accuracy_reward(completions, solution, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        #local_rank = int(os.environ.get("LOCAL_RANK", -1))
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content, sol in zip(contents, solution):
            # extract prediction
            prediction = parse(
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

        return rewards

    return accuracy_reward


def semi_accuracy_reward(completions, problem, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
        #print("RANK: {}\n, PROBLEM: {}\n, COMPLETION: {}\n".format(local_rank, problem[0], contents[0]))
    else:
        contents = [completion[0]["content"] for completion in completions]
    rewards = []
    predictions = []
    
    for content in contents:
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
        predictions.append(answer_parsed)

    common, max_similarity = find_most_similar_string(predictions, verify)

    for prediction in predictions:
        if len(common) != 0 and max_similarity > 1:
            reward = float(verify(common, prediction))
        else:
            reward = 0.0
        rewards.append(reward)
    
    return rewards

    

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]



def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]



def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solution

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards



def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward



def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
    

def find_most_similar_string(predictions, verify):
    max_similar_sum = 0  # 用于记录最大相似性总和
    most_similar_string = ''  # 用于记录最相似的字符串

    # 遍历每个字符串，计算与其他字符串的相似性总和
    for i in range(len(predictions)):
        current_string = predictions[i]
        total_similarity = 0

        # 计算当前字符串与其他字符串的相似性总和
        for j in range(len(predictions)):
            similarity = float(verify(current_string, predictions[j]))
            total_similarity += similarity  # 累加相似性值

        # 更新最相似的字符串和最大相似性总和
        if total_similarity > max_similar_sum:
            max_similar_sum = total_similarity
            most_similar_string = current_string

    return (most_similar_string, max_similar_sum)


def get_empo_math_reward(num_generations):
    def semantic_entropy_math_reward(completions, problem, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        all_contents = [completion[0]["content"] for completion in completions]
        all_rewards = []

        for i in range(0,len(all_contents), num_generations):
            contents=all_contents[i:i+num_generations]

            rewards = []
            predictions = []
        
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
        
            semantic_ids = get_semantic_ids_by_rule(predictions, rule=verify)
            n_generations = len(semantic_ids)
            counts = np.bincount(semantic_ids)
            probabilities = counts/n_generations
            assert np.isclose(probabilities.sum(), 1)
            total_entropy = -(probabilities * np.log(probabilities)).sum()
            #max_prob = np.max(probabilities)
            #max_prob_indices = np.where(probabilities == max_prob)[0]
        
            for index in range(len(contents)):
                # entropy thresholding to filter out highly unreliable answers
                if total_entropy < math.log(n_generations):
                    reward = math.log(probabilities[semantic_ids[index]])
                    rewards.append(reward)
                else:
                    rewards.append(0.0)
        
            all_rewards.extend(rewards)
        #print("RANK: {}, Contents: {}, Probability: {}, Semantic ID: {}, Reward: {}".format(local_rank, contents, probabilities, semantic_ids, rewards))
        return all_rewards
        
    return semantic_entropy_math_reward

    
def total_entropy_reward(completions, problem, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
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

#verifier = EntailmentDeberta()
verifier = GeneralVerifier()

def normalize_prediction(final_answer):
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    #final_answer = final_answer.split("=")[-1]

    #for before, after in SUBSTITUTIONS:
    #    final_answer = final_answer.replace(before, after)
    #for expr in REMOVED_EXPRESSIONS:
    #    final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    #final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    #final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    #final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    #final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    #final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    #final_answer = final_answer.replace("$", "")
    #final_answer = re.sub(r"^(0+)(?=[1-9])", "", final_answer)
    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    #final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    #final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    #final_answer = final_answer.replace("$", "")
    #final_answer = re.sub(r"^(0+)(?=[1-9])", "", final_answer)
    # Normalize 100,000 -> 100000
    #if final_answer.replace(",", "").isdigit():
    #    final_answer = final_answer.replace(",", "")

    return final_answer

def get_empo_common_reward(print_outputs=False):
    def semantic_prob_reward(completions, question, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        """Reward function that checks if the completion is the same as the ground truth."""
        if isinstance(completions[0], str):
            contents = [completion for completion in completions]
            #print("RANK: {}\n, PROBLEM: {}\n, COMPLETION: {}\n".format(local_rank, problem[0], contents[0]))
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
            #print(result)
            predictions.append(prediction)
            lengths.append(len(verifier.tokenizer.tokenize(prediction)))
            #result = normalize_prediction(result)
                
            #predictions.append(result)
        #prediction_with_context = ["The answer of question: \"{}\" is \"{}\"".format(q, a) if len(a)>0 else '' for q, a in zip(question, predictions)]
        #contents = predictions
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
        #for index in range(len(contents)):
        #    if total_entropy < math.log(len(probability)) and semantic_ids[index] in max_prob_indices:
        #        rewards.append(1.0)
        #    elif len(probabilities) == 1:
        #        rewards.append(1.0)
        #    else:
        #        rewards.append(0.0)
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
            #elif total_entropy < 0.57:
            #    rewards.append(max_prob)
            else:
                reward = probabilities[semantic_ids[index]] #* normalized_lengths[index]
                rewards.append(reward)

        if print_outputs:
            print("RANK: {},\n Question: {},\n Output: {}, Answers: {},\n Probability: {},\n Semantic ID: {}, \n Reward: {}\n\n".
                  format(local_rank, question[0], contents[0], predictions, probabilities, semantic_ids, rewards))
    
        return rewards

    return semantic_prob_reward

def exact_match_reward(completions, question, answer, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    """Reward function that checks if the completion is the same as the ground truth."""
    aliases = answer[0]['normalized_aliases']
    if isinstance(completions[0], str):
        contents = [completion for completion in completions]
        #print("RANK: {}\n, PROBLEM: {}\n, COMPLETION: {}\n".format(local_rank, problem[0], contents[0]))
    else:
        contents = [completion[0]["content"] for completion in completions]
    predictions = []
    # extract content in box
    for index, content in enumerate(contents):
        matches = re.findall(r"\\boxed\{(.*)\}", content)
        # 如果有多个 \boxed{}，取最后一个
        if matches:
            result = matches[-1]
        else:
            result = ""
        predictions.append(result.lower())
        
    rewards = [0] * len(contents)
    
    for index in range(len(contents)):
        for alias in aliases:
            if alias in predictions[index]:
                rewards[index] = 1.0
                break

    #print("RANK: {}, Predictions: {}, Answers: {}, Reward: {}".format(local_rank, predictions, aliases, rewards))
    
    return rewards


def get_general_accuracy_reward():
    def accuracy_reward(completions, reference_answer, question, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
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
            #print(result)
            predictions.append(prediction)
            #gold_answers.append(reference_answer)

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
