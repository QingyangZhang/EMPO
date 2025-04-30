### Requirements
You can install the required packages with the following command:
```bash
conda create -n math_eval python=3.10.9
conda activate math_eval
cd latex2sympy
pip install -e .
cd ..
pip install vllm==0.5.1 --no-build-isolation
pip install -r requirements.txt 
pip install transformers==4.45.0
```

### Evaluation
You can evaluate Qwen2.5/Qwen2-Math-Instruct series model with the following command:
```bash
# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES=0
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
OUTPUT_DIR="Qwen2.5-Math-1.5B-Instruct-Math-Eval"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR
```

## Acknowledgement
The evaluation codebase is borrowed from [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason).
