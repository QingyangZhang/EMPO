import re
import json
import os
from datasets import Dataset
import pandas as pd
import argparse
from jinja2 import Template
from sklearn.model_selection import train_test_split


def process_single_data(example, data_source, use_id):
    question = example['question']
    answer = example['answer'] if isinstance(example['answer'], str) else example['answer'][-1]
    if use_id:
        idx = example['id']
    else:
        idx = 0
    # answers = extract_answer(answer_raw)
    # if len(answers) == 0:
    #     raise ValueError("No boxed answer found")
    # if len(answers) != len(question_type):
    #     raise ValueError(f"Number of answers {len(answers)} does not match number of question types {len(question_type)}")
    
    system_prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    )

    user_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}. "
        "If the problem requires multiple answers to be answered, place all final answers in one \\boxed{} environment, separated by commas."
        "If the problem is in Chinese, provide your reasoning and answer in Chinese. Otherwise, use English."
        "{{prompt}}"
    )
    assistant_prompt = "Let me solve this step by step.\n<think>"

    user_prompt = Template(user_prompt).render(prompt=question)

    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "system",
            "content": system_prompt,
        },{
            "role": "user",
            "content": user_prompt
        }, {
            "role": "assistant",
            "content": assistant_prompt
        }],
        "ability": "Math",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(answer)
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "question": question
        }
    }
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/fs-computility/mabasic/shared/data/p1/train/PHYSICS-train-single-boxed.jsonl')
    parser.add_argument('--output_dir', type=str, default='/fs-computility/mabasic/shared/data/physics/sft_rl_data')
    parser.add_argument('--data_source', type=str, default='PHYSICS')
    parser.add_argument('--test_size', type=float, default=None)
    parser.add_argument('--use_id', action='store_true')

    args = parser.parse_args()

    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    processed_data = []
    for idx, example in enumerate(data):
        try:
            single_data = process_single_data(example, args.data_source, args.use_id)
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue
        processed_data.append(single_data)
    
    # 根据8:2随机划分训练集和测试集
    if args.test_size:
        train_data, test_data = train_test_split(processed_data, test_size=args.test_size, random_state=42)
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(args.output_dir, f'{args.data_source}_test.parquet'), engine='pyarrow', index=False)
    else:
        train_data = processed_data


    train_df = pd.DataFrame(train_data)

    train_df.to_parquet(os.path.join(args.output_dir, f'{args.data_source}_train.parquet'), engine='pyarrow', index=False)

    print("训练集和测试集已保存为 Parquet 格式。")
