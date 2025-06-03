from datasets import load_dataset
import re
from collections import defaultdict
import csv

# 加载数据集
dataset = load_dataset('hendrydong/gpqa_main_mc')['test']

# 初始化按领域分类的数据存储
domain_data = defaultdict(list)

for example in dataset:
    problem = example['problem']
    solution = example['solution']
    domain = example['domain']
    
    # 查找第一个(A)的位置分割问题
    a_pos = problem.find('(A)')
    if a_pos == -1:
        continue
    
    # 分割问题和选项部分
    question_part = problem[:a_pos].strip()
    options_part = problem[a_pos:].strip()

    target_suffix = r"Please write your final answer in the form of \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}"

    
    if options_part.endswith(target_suffix):
        options_part = options_part[:-len(target_suffix)].strip()

    print(options_part)

    text = options_part
    
    # 按选项标记分割文本（兼容多行内容）
    split_points = []
    current_pos = 0

    # 查找所有选项标记的位置（如 "(A)", "(B)"）
    while True:
        start = text.find('(', current_pos)
        if start == -1:
            break
        end = text.find(')', start)
        if end == -1:
            break
        split_points.append((start, end))  # 记录选项标记的位置
        current_pos = end + 1

    options_dict = {}

    # 提取每个选项内容（兼容多行）
    for i in range(len(split_points)):
        option_mark = text[split_points[i][0]+1 : split_points[i][1]]  # 提取字母（如 "A"）
    
        # 内容范围：从当前选项标记结束到下一个选项标记开始（或文本末尾）
        content_start = split_points[i][1] + 1
        content_end = split_points[i+1][0] if i < len(split_points)-1 else len(text)
    
        option_content = text[content_start : content_end].strip()  # 提取并清理内容
        options_dict[option_mark] = option_content

    print(options_dict)

    option_dict = options_dict

    #exit(0)

    # 验证选项完整性
    if len(option_dict) != 4 or any(k not in option_dict for k in ['A','B','C','D']):
        continue
    
    # 按字母顺序排列选项
    sorted_options = [option_dict[k] for k in ['A','B','C','D']]
    
    # 提取正确答案
    correct_match = re.search(r'\\boxed{([A-D])}', solution)
    if not correct_match:
        continue
    
    # 构建数据行
    row = [
        question_part,
        *sorted_options,
        correct_match.group(1)
    ]
    
    domain_data[domain].append(row)

# 生成CSV文件（保持不变）
for domain, rows in domain_data.items():
    filename = f"{domain.lower()}_test.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        #writer.writerow(['Problem', 'Option1', 'Option2', 'Option3', 'Option4', 'Correct'])
        writer.writerows(rows)

print("处理完成！生成文件列表：", [f"{k.lower()}_test.csv" for k in domain_data.keys()])