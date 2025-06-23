accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=7 src/open_r1/empo.py \
    --config recipes/Qwen2.5/rl/config_EMPO_NM_COT_20K_1.5B.yaml \
    --per_device_train_batch_size=1 --num_train_epochs=2

CUDA_VISIBLE_DEVICES=4,5,6,7 python occupy.py