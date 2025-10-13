# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"

export CUDA_VISIBLE_DEVICES=0
MODEL_NAME_OR_PATH="/mnt/shared-storage-user/zhangqingyang/ckpts/EMPO-Qwen2.5-Math-1.5B-step-384-hf"
OUTPUT_DIR="Qwen2.5-1.5B-EMPO-NM-COT-20K"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR