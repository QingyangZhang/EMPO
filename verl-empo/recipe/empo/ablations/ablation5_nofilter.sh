#!/usr/bin/env bash
set -x

export WANDB_API_KEY="5b84cdae0a11f6278f6e574a5ea1646dc33e1f7c"

project_name='PHYSICS-R1-DAPO'
exp_name='Ablation5-nofilter'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.20

max_prompt_length=$((1024))
max_response_length=$((16384))
enable_overlong_buffer=False
overlong_buffer_len=1024
overlong_penalty_factor=0.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=32
n_resp_per_prompt=8

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/fs-computility/mabasic/shared"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/p1/sft/distill_v3"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/data/physics/ckpts/${project_name}/${exp_name}"}
PHYSICS_train_path=${RAY_DATA_HOME}/data/physics/sft_rl_data/PHYSICS_train.parquet
ugphysics_train_path=${RAY_DATA_HOME}/data/physics/sft_rl_data/ugphysics_train.parquet
scpphysics_train_path=${RAY_DATA_HOME}/data/physics/sft_rl_data/scp-physics_train.parquet
math_test_path=${RAY_DATA_HOME}/data/physics/sft_rl_data/math_test.parquet
gpqa_test_path=${RAY_DATA_HOME}/data/physics/sft_rl_data/gpqa_phy.parquet
OlympiadBench_test_path=${RAY_DATA_HOME}/data/physics/sft_rl_data/OlympiadBench.parquet
TRAIN_FILES="['$ugphysics_train_path','$scpphysics_train_path', '$PHYSICS_train_path']"
TEST_FILES="['$gpqa_test_path','$OlympiadBench_test_path']"


# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False

python3 -m recipe.physics.src.main_dapo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${MLP_WORKER_GPU:-1}" \
    trainer.nnodes="${MLP_WORKER_NUM:-1}" \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=100 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable 2>&1 | tee outputs/${exp_name}-6.log