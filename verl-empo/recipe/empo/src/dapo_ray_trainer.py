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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict, Counter
from copy import deepcopy
from pprint import pprint
import wandb
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_empo_metrics,
    compute_overlong_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.debug import marked_timer


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        
        # load checkpoint before doing anything
        self._load_checkpoint()
            

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate(logger)
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        is_best_step = False
        best_val_metrics = 0

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        prompt_uid2scores = defaultdict(lambda: {'scores': [], 'sources': str})
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # empo batch info
                new_batch.meta_info["do_vote"] = False
                if self.use_empo:
                    self.config.actor_rollout_ref.rollout.n = self.n_votes_per_prompt
                    new_batch.meta_info["do_vote"] = True

                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                n_samples = self.config.actor_rollout_ref.rollout.n
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=False)
                gen_batch.meta_info['is_train'] = True
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        indices = torch.arange(len(gen_batch_output)).reshape(self.config.actor_rollout_ref.rollout.n, -1).T.reshape(-1)
                        gen_batch_output.reorder(indices)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # recorde the scores(rollout n times) for each prompt uid
                        for uid, score, source in zip(new_batch.non_tensor_batch['uid'], new_batch.batch['token_level_scores'].sum(-1), new_batch.non_tensor_batch['data_source']):
                            prompt_uid2scores[uid]['scores'].append(score)
                            prompt_uid2scores[uid]['sources'] = source
                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        prompt_uid2metric_max = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            numeric_metric_vals = [float(val) for val in metric_vals]
                            prompt_uid2metric_std[prompt_uid] = np.std(numeric_metric_vals)
                            prompt_uid2metric_max[prompt_uid] = np.max(numeric_metric_vals)
                        
                        lower = self.config.algorithm.filter_groups.confidence_lower_bound
                        upper = self.config.algorithm.filter_groups.confidence_upper_bound
                        kept_prompt_uids = [uid for uid, max in prompt_uid2metric_max.items() if (max >= lower and max <= upper) or len(prompt_uid2metric_vals[uid]) == 1]
                        
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"num_prompt_in_batch={num_prompt_in_batch}, {num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]
                    
                    sample_size = min(len(batch), 8)

                    if self.use_empo:
                        table_data = [
                            [
                                batch.non_tensor_batch['index'].tolist()[i],
                                batch.non_tensor_batch['prompt'].tolist()[i],
                                batch.non_tensor_batch['response'].tolist()[i],
                                batch.non_tensor_batch['extracted_pred'].tolist()[i],
                                batch.non_tensor_batch['estimated_label'].tolist()[i],
                                batch.non_tensor_batch['extracted_gt'].tolist()[i], 
                                batch.non_tensor_batch['score'].tolist()[i],
                            ]
                            for i in range(sample_size)
                        ]
                    
                        logger.log({
                            "generated_examples": wandb.Table(
                                columns=['index', 'prompt', 'response', 'extracted_answer', 'pseudo_label', 'gold_answer', 'raw_reward'],
                                data=table_data,
                            )
                        }, step=self.global_steps)
                    else:
                        table_data = [
                            [
                                batch.non_tensor_batch['prompt'].tolist()[i],
                                batch.non_tensor_batch['response'].tolist()[i],
                                batch.non_tensor_batch['extracted_pred'].tolist()[i],
                                batch.non_tensor_batch['extracted_gt'].tolist()[i],
                                batch.non_tensor_batch['score'][i],
                                batch[i].non_tensor_batch['data_source'],
                                batch[i].non_tensor_batch['scored_by']
                            ]
                            for i in np.random.choice(len(batch), sample_size, replace=False)
                        ]
                        logger.log({
                            "generated_examples": wandb.Table(
                                columns=['prompt', 'completion', 'extracted_answer', 'gold_answer', 'raw_reward', 'data_source', 'scored_by'],
                                data=table_data,
                            )
                        }, step=self.global_steps, backend='wandb')
                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        batch.batch["token_level_entropys"] = entropys
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate(logger)
                        metrics.update(val_metrics)

                    if is_best_step:
                        with marked_timer("save_best_checkpoint", timing_raw, "green"):
                            self._save_checkpoint(is_best_step=True)
                            is_best_step = False

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            # self._save_checkpoint()
                            self._save_checkpoint(is_best_step=False)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                if self.use_empo:
                    metrics.update(compute_empo_metrics(batch=batch))

                if "overlong_reward" in batch.non_tensor_batch.keys():
                    metrics.update(compute_overlong_metrics(batch=batch))
                
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1