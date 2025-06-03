export CUDA_VISIBLE_DEVICES=0,1,2,3

# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-7B-Instruct \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --use_vllm \
'''

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-7B-EMPO-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-Instruct-EMPO-natural_reasoning_simple_from_base_general-verifier   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-Instruct-EMPO-natural_reasoning_simple_from_base_general-verifier   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-7B-base-1-shot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B   \
    --eval_batch_size 32 \
    --ntrain 1 \
    --cot \
    --use_vllm \
'''
'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-3B-SFT-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-SFT-NR   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-SFT-NR   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''


'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-3B-Instruct-0-shot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''
    
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-3B-Random-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-Random-Natural-Reasoning   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-Random-Natural-Reasoning   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-3B-GRPO-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-GRPO-Natural-Reasoning   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-GRPO-Natural-Reasoning   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-3B-EMPO-1-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-EMPO-Natural-Reasoning-from-base-0421   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-EMPO-Natural-Reasoning-from-base-0421   \
    --eval_batch_size 32 \
    --ntrain 1 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-3B-1-shot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B   \
    --eval_batch_size 32 \
    --ntrain 1 \
    --cot \
    --use_vllm \
'''

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-7B-GRPO-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-GRPO-Natural-Reasoning-0428   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-GRPO-Natural-Reasoning-0428   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-14B-Instruct-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''
'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-14B-GRPO-Natural-Reasoning \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-GRPO-Natural-Reasoning   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-GRPO-Natural-Reasoning   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''
'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-14B-base-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
'''

'''
python -m eval.gpqa.run_eval \
    --data_dir data/eval/gpqa \
    --save_dir results/GPQA/Qwen2.5-14B-SFT-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-SFT-NR   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-SFT-NR   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
'''