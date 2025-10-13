# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score.empo.compute_metrics import empo_metrics

from verl.workers.reward_manager import register

import traceback
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from verl.utils.reward_score import _default_compute_score, empo_compute_score
import psutil
from tqdm import tqdm

async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor):
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            executor,
            partial(evaluation_func, task, completion, reference, task_extra_info)
        )
        return await future
    except:
        e = traceback.format_exc()
        print(f"[Error] Fuction `single_compute_score` failed: {e}, completion: {completion[-80:]}")
        return None


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes, timeout):
    if extra_info is None:
        extra_info = [None] * len(tasks)

    results = [None] * len(tasks)

    futures = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i, (c, r, t, ei) in enumerate(zip(completions, references, tasks, extra_info)):
            future = asyncio.create_task(
                single_compute_score(evaluation_func, c, r, t, ei, executor)
            )
            futures.append((i, future))

        try:
            total_tasks = len(futures)
            pbar = tqdm(total=total_tasks, desc="Computing scores", unit="task")
            
            for i, future in futures:
                try:
                    result = await asyncio.wait_for(future, timeout=timeout)
                    results[i] = result
                except asyncio.TimeoutError:
                    print(f"[Timeout] Reward future {i} timeout after {timeout}s")
                    results[i] = None
                except Exception:
                    e = traceback.format_exc()
                    print(f"[Error] Reward future {i} failed: {e}")
                    results[i] = None
                finally:
                    pbar.update(1)
            
            pbar.close()
            print("[Success] All tasks gathered.")
        except Exception as e:
            print(f"[Exception] in processing futures: {e}")
        finally:
            print("[Shutdown] Cleaning up remaining subprocesses...")
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    formatted = []
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            formatted.append({
                "score": 0.,
                "point": 0.,
                "acc": False,
                "extracted_gt": str(reference),
                "extracted_pred": 'Reward Exception',
                "scored_by": "not_scored"
            })
        elif isinstance(result, dict):
            formatted.append(result)
        else:
            formatted.append(result[0])
    return formatted

def run_reward_scoring(compute_score_func, completions, references, tasks, extra_info, num_processes=64, timeout=300.):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(parallel_compute_score_async(
            compute_score_func, completions, references, tasks, extra_info, 
            num_processes, timeout
        ))
    finally:
        loop.close()

@register("empo")
class EMPORewardManager:
    """The reward manager."""

    def __init__(self, 
                tokenizer, 
                num_examine,
                compute_score=None, 
                reward_fn_key="data_source",
                max_resp_len=None,
                overlong_buffer_cfg=None, 
                n_votes_per_prompt=1, 
                n_samples_per_prompt=1, 
                mode="eval",
                eval_n_samples=1, 
                reward_type=None, 
                entropy_thres=None,
                use_xverify=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = empo_compute_score # _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        self.reward_type = reward_type
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.entropy_thres = entropy_thres
        self.compute_score = partial(self.compute_score, use_xverify=use_xverify)

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

        print(f"EMPORewardManager initialized with n_votes_per_prompt {n_votes_per_prompt}, n_samples_per_prompt {n_samples_per_prompt}, eval_n_samples {eval_n_samples}, reward_type {reward_type}, entropy_thres {entropy_thres}")

    def _compute_eval_reward(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # batched scoring
        response_ids = data.batch['responses']
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = data.non_tensor_batch.get('extra_info', None)

        assert len(responses_str) == len(ground_truths) == len(data_sources) == len(extra_infos)
        # Process in batches of 500
        batch_size = 500
        results = []
        
        for i in range(0, len(responses_str), batch_size):
            batch_responses = responses_str[i:i+batch_size]
            batch_ground_truths = ground_truths[i:i+batch_size] 
            batch_data_sources = data_sources[i:i+batch_size]
            batch_extra_infos = extra_infos[i:i+batch_size] if extra_infos is not None else None
            try:
                batch_results = run_reward_scoring(
                    self.compute_score,
                    completions=batch_responses,
                    references=batch_ground_truths,
                    tasks=batch_data_sources,
                    extra_info=batch_extra_infos,
                    num_processes=64,
                    timeout=300.,
                )
                results.extend(batch_results)
            except OverflowError as e:
                print(f"OverflowError in batched reward computing. Setting all as 0.: {e}")
                results = [{
                    "score": 0.,
                    "point": 0.,
                    "acc": False,
                    "extracted_gt": str(gt),
                    "extracted_pred": None,
                    "scored_by": "not_scored"
                } for gt in batch_ground_truths]
            except asyncio.TimeoutError as e:
                print('Global timeout in reward computing! Setting all as 0.')
                results = [{
                    "score": 0.,
                    "point": 0.,
                    "acc": False,
                    "extracted_gt": str(gt),
                    "extracted_pred": None,
                    "scored_by": "not_scored"
                } for gt in batch_ground_truths]
            except Exception:
                e = traceback.format_exc()
                print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
                results = [{
                    "score": 0.,
                    "point": 0.,
                "acc": False,
                "extracted_gt": str(gt),
                "extracted_pred": 'Reward Exception',
                "scored_by": "not_scored"
            } for gt in ground_truths]
        
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

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('**************************************')
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
            reward_extra_info['prompt'].append(prompt_str)
            reward_extra_info['response'].append(response_str)

        return reward_tensor, reward_extra_info, None

    def __call__(self, data: DataProto, return_dict=False):

        if self.mode == "train":
            reward_tensor, reward_extra_info, empo_info = self._compute_empo_reward(data)
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, empo_info = self._compute_eval_reward(data)
        else:
            raise NotImplementedError(f"Mode {self.mode} is not supported for EMPORewardManager")

        if return_dict:
            return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                    "empo_info": empo_info,
                }
        else:
            return reward_tensor

    async def _single_empo_task(self, group_args, executor):
        """
        此函数现在将直接执行任务，并将超时处理留给调用方（_compute_empo_reward_async）。
        """
        group_pred_outputs, group_labels, task, group_extra_info, reward_type, entropy_thres = group_args
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(
                executor,
                partial(empo_metrics, group_pred_outputs, group_labels, group_extra_info, reward_type, entropy_thres)
            )
            return await future
        except Exception:
            e = traceback.format_exc()
            index = group_extra_info[0].get('index', 'N/A')
            print(f"[Error] Function `empo_metrics` failed for group {index}: {e}")
            return None, {}

    async def _compute_empo_reward_async(self, data: "DataProto"):
        assert len(data) % self.n_votes_per_prompt == 0, f"Length of data {len(data)} must be divisible by n_votes_per_prompt {self.n_votes_per_prompt}"
        prompt_num = len(data) // self.n_votes_per_prompt
        reward_tensor = torch.zeros_like(data.batch["responses"][:prompt_num * self.n_samples_per_prompt], dtype=torch.float32)
        scores = [0.0] * len(data)
        prompts_str_set = set()
        reward_extra_info = defaultdict(list)
        all_empo_metrics = defaultdict(list)
        already_print_data_sources = {}
        
        with ProcessPoolExecutor(max_workers=64) as executor:
            tasks_to_process = []
            # Loop 1: create
            for prompt_i in range(prompt_num):
                group_pred_outputs, group_labels, group_extra_info = [], [], []
                task_type = None
                results_template = []
                for i in range(self.n_votes_per_prompt):
                    data_item = data[prompt_i * self.n_votes_per_prompt + i]
                    # --- 完整的数据提取逻辑 ---
                    prompt_idx = data_item.batch["prompts"]
                    prompt_length = prompt_idx.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                    response_idx = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_idx = response_idx[:valid_response_length]
                    prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    extra_info = data_item.non_tensor_batch["extra_info"]
                    if isinstance(ground_truth, list):
                        ground_truth = ground_truth[-1]
                    group_labels.append(ground_truth)
                    group_pred_outputs.append(response_str)
                    group_extra_info.append(extra_info)
                    results_template.append({
                        "prompt": prompt_str, "response": response_str, "extracted_pred": "",
                        "extracted_gt": ground_truth, "estimated_label": ground_truth,
                        "score": 0., "data_source": data_source, "scored_by": "not_scored",
                        "index": extra_info.get("index", 'N/A'), "valid_response_length": valid_response_length,
                        "label_accuracy": 0., "majority_ratio": 0., f"pass@{self.n_votes_per_prompt}": 0.,
                    })
                    prompts_str_set.add(prompt_str)

                task_args = (group_pred_outputs, group_labels, task_type, group_extra_info, self.reward_type, self.entropy_thres)
                # 传入 executor 给 _single_empo_task，以便它可以在进程池中运行
                async_task = asyncio.create_task(self._single_empo_task(task_args, executor))
                tasks_to_process.append({
                    "task": async_task,
                    "prompt_i": prompt_i,
                    "template": results_template,
                    "index": group_extra_info[0].get('index', 'N/A')
                })

            # 循环 2: 等待任务完成并立即处理结果
            try:
                for task_info in tqdm(tasks_to_process, desc="Computing EMPO rewards"):
                    try:
                        # 仅在这一层进行超时控制
                        rewards, empo_reward_metrics = await asyncio.wait_for(task_info["task"], timeout=600.0) # 外部总超时
                    except asyncio.TimeoutError:
                        print(f"[Timeout] Task for group {task_info['index']} timed out after 600.0s.")
                        rewards, empo_reward_metrics = None, {}
                        # 尽管这里捕获了超时，但底层的进程可能仍在运行，需要 finally 块清理
                    except Exception as e:
                        print(f"[Error] Task for group {task_info['index']} failed unexpectedly: {e}")
                        rewards, empo_reward_metrics = None, {}

                    # --- 结果处理逻辑 ---
                    prompt_i = task_info['prompt_i']
                    results_template_for_group = task_info['template']

                    if rewards is None:
                        rewards = [0.0] * len(results_template_for_group)
                        estimated_labels = [r["extracted_gt"] for r in results_template_for_group]
                        model_answers = ["Error in reward computation"] * len(results_template_for_group)
                        empo_reward_metrics = {}  # 确保在失败时这是一个空字典
                        estimated_labels = [''] * len(results_template_for_group)
                        model_answers = [''] * len(results_template_for_group)
                        label_accuracies = [0.0] * len(results_template_for_group)
                        majority_ratios = [0.0] * len(results_template_for_group)
                    else:
                        estimated_labels = empo_reward_metrics.pop("estimated_label", [r["extracted_gt"] for r in results_template_for_group])
                        model_answers = empo_reward_metrics.pop("extracted_answers", [r["response"] for r in results_template_for_group])
                        label_accuracies = empo_reward_metrics.pop("label_accuracy", [r["label_accuracy"] for r in results_template_for_group])
                        majority_ratios = empo_reward_metrics.pop("majority_ratio", [r["majority_ratio"] for r in results_template_for_group])

                    for i, (result_template, model_answer, estimated_label, label_accuracy, majority_ratio, reward) in enumerate(zip(results_template_for_group, model_answers, estimated_labels, label_accuracies, majority_ratios, rewards)):
                        result_template.update({
                            "estimated_label": estimated_label,
                            "extracted_pred": model_answer,
                            "label_accuracy":  label_accuracy,
                            "majority_ratio": majority_ratio,
                            "score": str(reward)
                        })

                        
                        for key, value in result_template.items():
                            if key != "valid_response_length":
                                reward_extra_info[key].append(value)
                        
                        score_index = prompt_i * self.n_votes_per_prompt + i
                        scores[score_index] = reward

                        if i < self.n_samples_per_prompt:
                            tensor_index = prompt_i * self.n_samples_per_prompt + i
                            reward_tensor[tensor_index, result_template["valid_response_length"] - 1] = reward
                        data_source = result_template["data_source"]
                        if data_source not in already_print_data_sources:
                            already_print_data_sources[data_source] = 0
                        if already_print_data_sources[data_source] < self.num_examine:
                            already_print_data_sources[data_source] += 1
                            print(f"[prompt] {result_template['prompt']}\n[response] {result_template['response']}\n[score] {reward}")

                    for k, v in empo_reward_metrics.items():
                        all_empo_metrics[k].append(v)
                print("[Success] All EMPO tasks gathered.")
            except Exception as e:
                print(f"[Exception] in processing EMPO futures: {e}")
            finally:
                # 确保在任何情况下都尝试清理子进程，无论任务是否超时或失败
                print("[Shutdown] Cleaning up remaining EMPO subprocesses...")
                terminated_count, total_count = 0, 0
                for pid, proc in list(executor._processes.items()): # 使用 list() 避免在迭代时修改字典
                    try:
                        p = psutil.Process(pid)
                        if p.is_running(): # 检查进程是否仍在运行
                            # print(f"Attempting to terminate EMPO process {pid}...")
                            p.terminate() # 发送 SIGTERM 信号
                            try:
                                p.wait(timeout=600.) # 等待进程优雅退出
                            except psutil.TimeoutExpired:
                                print(f"EMPO process {pid} did not terminate gracefully, killing...")
                                p.kill() # 强制终止 SIGKILL
                            terminated_count += 1
                    except psutil.NoSuchProcess:
                        # 进程可能已经自行退出
                        pass
                    except Exception as ex:
                        print(f"Error while cleaning up EMPO process {pid}: {ex}")
                    finally:
                        total_count += 1
                print(f"[Shutdown] {terminated_count} of {total_count} EMPO subprocess(es) terminated.")


        # --- 最终处理和返回 ---
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
        empo_info = {}

        assert len(reward_extra_info['prompt']) == len(data), f"Incomplete reward info, expect {len(data)} but got {len(reward_extra_info['prompt'])}"
        assert len(reward_extra_info['label_accuracy']) == len(data), f"Incomplete EMPO info, expect {len(data)} but got {len(reward_extra_info['label_accuracy'])}"
        if len(prompts_str_set) < prompt_num:
            print(f"[Warning] Expected {prompt_num} distinct prompts but got {len(prompts_str_set)}, which is abnormal except for TTT setting.")

        return reward_tensor, reward_extra_info, empo_info

    def _compute_empo_reward(self, data: "DataProto"):
        """
        同步入口点。
        使用独立的事件循环来运行异步代码，这是在库或框架（如Ray）中推荐的做法。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._compute_empo_reward_async(data))
        finally:
            loop.close()