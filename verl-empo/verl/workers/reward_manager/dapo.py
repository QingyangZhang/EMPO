import traceback
from functools import partial
import torch
from verl import DataProto
from verl.utils.reward_score import _default_compute_score, empo_compute_score
from collections import defaultdict
from tqdm import tqdm

from verl.workers.reward_manager import register

@register("dapo")
class DAPORewardManager:
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 max_resp_len=None,
                 overlong_buffer_cfg=None,
                 use_xverify=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or empo_compute_score # _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.compute_score = partial(self.compute_score, use_xverify=use_xverify)

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Prepare data for scoring
        response_ids = data.batch['responses']
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        # Ensure extra_infos has a value for each item, even if it's None
        extra_infos = data.non_tensor_batch.get('extra_info', [None] * len(data))

        assert len(responses_str) == len(ground_truths) == len(data_sources) == len(extra_infos)
        
        # Simplified serial execution loop
        results = []
        print("Computing scores sequentially...")
        iterator = zip(responses_str, ground_truths, data_sources, extra_infos)
        for completion, reference, task, extra_info in tqdm(iterator, total=len(responses_str), desc="Computing scores"):
            try:
                # Directly call the scoring function
                result = self.compute_score(
                    task,
                    completion,
                    reference,
                    extra_info
                )
                results.append(result)
            except Exception:
                e = traceback.format_exc()
                print(f"Unexpected error in reward computing. Setting score to 0.: {e}")
                # Append a default error result to maintain data structure
                error_result = {
                    "score": 0.,
                    "point": 0.,
                    "acc": False,
                    "extracted_gt": str(reference),
                    "extracted_pred": 'Reward Exception',
                    "scored_by": "not_scored"
                }
                results.append(error_result)

        # Process the collected results
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            result = results[i]

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            assert not isinstance(score, list), f"{score=} is list"
            reward = score

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # Ensure there's a valid index to place the reward
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('**************************************')
                print("[ground_truth]", ground_truth)
                # print("[response]", response_str)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
            reward_extra_info['prompt'].append(prompt_str)
            reward_extra_info['response'].append(response_str)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor