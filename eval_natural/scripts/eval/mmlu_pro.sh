# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0,1,2,3


'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-Instruct \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --eval_batch_size 16 \
    --ntrain 0 \
    --cot
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --use_vllm \
'''

python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-Random-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-Random-Natural-Reasoning   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-Random-Natural-Reasoning   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-cot-0-shot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --cot \
    --use_vllm \
'''
'''
python -m eval.mmlu.run_eval \
    --data_dir data/eval/mmlu \
    --save_dir results/MMLU/Qwen2.5-3B-Instruct-cot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''
'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-Instruct-cot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''
'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-7B-EMPO-cot-simple-full \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-Instruct-EMPO-natural_reasoning_simple_full   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-Instruct-EMPO-natural_reasoning_simple_full   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''
'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-7B-Instruct \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''
'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-Instruct-cot-0-shot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-3B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-7B-EMPO-cot-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-Instruct-EMPO-natural_reasoning_simple_from_base_general-verifier \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-Instruct-EMPO-natural_reasoning_simple_from_base_general-verifier \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-7B-cot-5-shot \
    --model /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B \
    --tokenizer /apdcephfs_qy3/share_1594716/bingzhe/pretrained/Qwen2.5-7B \
    --eval_batch_size 32 \
    --ntrain 5 \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''


'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-EMPO-5-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-EMPO-Natural-Reasoning-STEM-20K-free-form   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-EMPO-Natural-Reasoning-STEM-20K-free-form   \
    --eval_batch_size 32 \
    --ntrain 5 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-7B-SFT-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-SFT-NR   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-SFT-NR   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-GRPO-Natural-Reasoning \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-GRPO-Natural-Reasoning   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-GRPO-Natural-Reasoning   \
    --eval_batch_size 64 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-7B-GRPO-Natural-Reasoning-0428-continue \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-GRPO-Natural-Reasoning-0428/Qwen2.5-7B-GRPO-Natural-Reasoning-0428-continue   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-GRPO-Natural-Reasoning-0428/Qwen2.5-7B-GRPO-Natural-Reasoning-0428-continue   \
    --eval_batch_size 1 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-3B-EMPO-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-EMPO-Natural-Reasoning-0506   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-3B-EMPO-Natural-Reasoning-0506   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''


'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-14B-EMPO-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-EMPO-Natural-Reasoning   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-EMPO-Natural-Reasoning   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \

'''


'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-14B-GRPO-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-GRPO-Natural-Reasoning   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-GRPO-Natural-Reasoning   \
    --eval_batch_size 4 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''

'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-14B-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''


'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-14B-SFT-NR-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-SFT-NR   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-14B-SFT-NR   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''
    
'''
python -m eval.mmlu_pro.run_eval \
    --data_dir data/eval/mmlu_pro \
    --save_dir results/MMLU_PRO/Qwen2.5-14B-Instruct-0-shot \
    --model /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B-Instruct   \
    --tokenizer /apdcephfs_qy3/share_1594716/yataobian/yang/pretrained/Qwen2.5-14B-Instruct   \
    --eval_batch_size 32 \
    --ntrain 0 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    --cot \
    --use_vllm \
'''