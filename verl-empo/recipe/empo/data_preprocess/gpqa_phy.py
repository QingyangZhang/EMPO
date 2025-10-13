import json
import os
from datasets import Dataset
import pandas as pd
import argparse
from jinja2 import Template


def process_single_data(example, idx):
    question = example['problem']
    answer = example['answer']

    system_prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    )

    user_prompt = (
        "{{prompt}}."
        "Please show your choice in the answer field with only the choice letter, e.g., \"answer": \"C\".\" "
    )

    user_prompt = Template(user_prompt).render(prompt=question)

    data = {
        "data_source": data_source,
        "prompt": [{
            "role": "system",
            "content": system_prompt,
        },{
            "role": "user",
            "content": user_prompt
        }],
        "ability": "physics",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            "split": "test",
            "index": idx,
            "question": question,
        }
    }
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='/fs-computility/mabasic/shared/data/p1/test/gpqa_phy_fixed.jsonl')
    parser.add_argument('--output_dir', type=str, default='/fs-computility/mabasic/shared/data/physics/rl_data')
    data_source = 'gpqa_phy'
    args = parser.parse_args()

    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    processed_data = [process_single_data(example, idx) for idx, example in enumerate(data)]
    df = pd.DataFrame(processed_data)
    dataset = Dataset.from_pandas(df)

    # 保存为 Parquet 文件
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.to_parquet(os.path.join(args.output_dir, 'gpqa_phy.parquet'))

    print(f"Processed data saved to {os.path.join(args.output_dir, 'gpqa_phy.parquet')}")

