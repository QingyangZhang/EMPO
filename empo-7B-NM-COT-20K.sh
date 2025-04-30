accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=7 src/open_r1/dapo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_EMPO_NM_COT_20K_7B.yaml \
    --per_device_train_batch_size=1 --num_train_epochs=1

CUDA_VISIBLE_DEVICES=3,4,5,6,7 python occupy.py