from functools import partial
from verl.utils.reward_score.empo_verify import solution2answer

def auto_extract(all_outputs, extra_info=None):
    
    extract_fn = partial(solution2answer)

    model_answers = [extract_fn(generated_text) for generated_text in all_outputs]

    extracted_answers = []

    for answer in model_answers:
        assert answer is not None
        if isinstance(answer, list):
            extracted_answers.append(answer[-1])
        else:
            extracted_answers.append(answer)

    return extracted_answers