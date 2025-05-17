accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=6 src/open_r1/empo.py \
    --config recipes/Qwen2.5/rl/config_EMPO_NR_50K_3B.yaml \
    --per_device_train_batch_size=1 --num_train_epochs=1