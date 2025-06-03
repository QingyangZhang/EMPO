import pandas as pd
import os
from ast import literal_eval

import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["validation"])
# 假设 df 是你的原始 DataFrame
# 如果 options 列是字符串格式的列表，需要先转换为实际的列表
# 如果已经是列表格式，可以跳过这一步
df['options'] = df['options'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

# 按 category 分组
grouped = df.groupby('category')

# 遍历每个分组并保存为 CSV 文件
for category, group in grouped:
    # 创建一个新的 DataFrame 用于保存
    output_df = pd.DataFrame()
    
    # 保留 question 列
    output_df['question'] = group['question']
    
    # 将 options 列展开为 10 个列
    options_expanded = group['options'].apply(pd.Series)
    # 确保有 10 列，如果不足则填充空值
    options_expanded = options_expanded.reindex(columns=range(10), fill_value='')
    output_df = pd.concat([output_df, options_expanded], axis=1)
    
    # 添加 answer 列作为最后一列
    output_df['answer'] = group['answer']
    
    # 保存为 CSV 文件
    filename = f"/apdcephfs_qy3/share_1594716/yataobian/yang/EMPO/open-instruct/data/eval/mmlu_pro/dev/{category}_dev.csv"
    output_df.to_csv(filename, index=False, header=False)
    print(f"Saved {filename} with {len(output_df)} rows")