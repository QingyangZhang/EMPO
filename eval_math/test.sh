# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES=1
MODEL_NAME_OR_PATH="/apdcephfs_qy3/share_1594716/yataobian/yang/output/data/Qwen2.5-7B-EMPO-AMC-17K-0404-3epoch"
OUTPUT_DIR="Qwen2.5-7B-EMPO-AMC-17K-0404-3epoch"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR