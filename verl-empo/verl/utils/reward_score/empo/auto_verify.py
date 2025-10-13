from collections import defaultdict
from verl.utils.reward_score.empo.auto_extract import auto_extract
from typing import List
from verl.utils.reward_score.empo_verify import solution2answer, grade_single_answer

def qwen_reward_fn_empo(extracted_text, golden_answer):
    model_answer = extracted_text
    if isinstance(model_answer, list):
        assert len(model_answer) == 1 and isinstance(model_answer[0], str), "model answer must be str or list with one element"
        model_answer = model_answer[0]
    accuracy = 1.0 if grade_single_answer(model_answer, golden_answer) else 0.0 #-0.5
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    # print(f"grade {model_answer}, {golden_answer}, got {grade_single_answer(model_answer, golden_answer)}")
    return accuracy

def auto_verify(all_outputs, all_labels, extra_info=None):
    verify_fn = qwen_reward_fn_empo
    verify_extra_info = defaultdict(list)

    assert (
        isinstance(all_outputs, list) and 
        all(isinstance(item, str) for item in all_outputs)
        ), (
        f"model answers must be list[str], "
        f"got {type(all_outputs).__name__} containing {[type(x).__name__ for x in all_outputs[:1]]}..."
        )

    assert (
        isinstance(all_labels, list) and 
        all(isinstance(item, str) for item in all_labels)
        ), (
        f"gts must be list[str], "
        f"got {type(all_labels).__name__} containing {[type(x).__name__ for x in all_labels[:1]]}..."
        )

    rewards = [verify_fn(output, label)
                   for output, label in zip(all_outputs, all_labels)]
    
    verify_extra_info["acc"] = rewards

    verify_extra_info["pred"] = auto_extract(all_outputs)
        
    return rewards, verify_extra_info

if __name__ == "__main__":
    # Example usage
    model_output = "The answer is <answer>$\\boxed{4.96}$</answer>"
    ground_truth = "[4.92, 4.97]"
    question = "What is the answer to the ultimate question of life, the universe, and everything?"
    
    model_answer = auto_extract(all_outputs=model_answer)
    result = auto_verify(all_outputs=[model_answer], all_labels=[ground_truth], questions=[question])

    print(result)