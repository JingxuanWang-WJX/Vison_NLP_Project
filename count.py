import json
from collections import defaultdict

def analyze_json_file(file_path):
    """
    分析JSON文件，统计总记录数、重复的image_path和生成失败的记录。

    Args:
        file_path (str): JSON文件的路径。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式。")
        return

    # 用于存储每个image_path出现的次数和位置
    image_path_counts = defaultdict(list)
    for i, record in enumerate(data):
        # 检查 image_path 是否存在
        if 'image_path' in record:
            image_path_counts[record['image_path']].append(i)

    # 找出重复的image_path
    duplicates = {path: indices for path, indices in image_path_counts.items() if len(indices) > 1}

    # 找出生成失败的记录
    failed_creations = []
    for i, record in enumerate(data):
        # 检查 create_condition 是否为 False
        if 'generated_content' in record and record['generated_content'].get('create_condition') is False:
            failed_creations.append({
                'index': i,
                'record': record
            })

    # --- 打印结果 ---
    print("--- JSON文件分析结果 ---\n")

    # 打印总记录数
    print(f"文件总记录数: {len(data)}\n")
    print("-" * 20)


    # 打印重复的 image_path 统计
    if duplicates:
        print(f"共有 {len(duplicates)} 个重复的 image_path，涉及 {sum(len(indices) for indices in duplicates.values())} 条记录。\n")
        print("重复的 image_path 及其对应的记录索引：")
        for path, indices in duplicates.items():
            print(f"  - 路径: {path}")
            print(f"    记录索引: {indices}")
            print(f"    重复次数: {len(indices)}")
        print("-" * 20)
    else:
        print("没有发现重复的 image_path。\n")

    # 打印生成失败的记录统计
    if failed_creations:
        print(f"共有 {len(failed_creations)} 条记录诗词生成失败。\n")
        print("生成失败的记录详情：")
        for item in failed_creations:
            print(f"  - 记录索引: {item['index']}")
            print(f"    Image Path: {item['record'].get('image_path', 'N/A')}")
            print(f"    失败原因: {item['record']['generated_content'].get('poem_explanation', '未提供')}")
        print("-" * 20)
    else:
        print("没有发现诗词生成失败的记录。\n")


# 调用函数，传入您的JSON文件名
# 请确保 'output_list.json' 文件与此脚本在同一目录下
analyze_json_file('output_list.json')
