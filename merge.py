import json

# 定义要合并的JSON文件名列表
# Define the list of JSON filenames to be merged
file_names = [
    'output_list_460_tem3.json',
    'output_list_470_tem1.json',
    'output_list_590_tem2.json',
    'output_list_110_tem4.json',
    'output_list_70_tem5.json',
    'output_list_130_tem6.json',
    'output_list_500_tem7.json',
    'output_list_tem8.json',
    'output_list_tem9.json'
]

# 创建一个空列表，用于存储所有合并后的数据
# Create an empty list to store all the merged data
merged_data = []

# 遍历文件名列表
# Iterate through the list of filenames
for file_name in file_names:
    try:
        # 打开并读取每个JSON文件
        # Open and read each JSON file
        with open(file_name, 'r', encoding='utf-8') as f:
            # 加载JSON数据并将其追加到合并列表中
            # Load the JSON data and extend the merged list with it
            data = json.load(f)
            merged_data.extend(data)
            print(f"成功读取并合并文件: {file_name}")
    except FileNotFoundError:
        print(f"错误: 文件 '{file_name}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_name}' 不是有效的JSON格式。")
    except Exception as e:
        print(f"处理文件 '{file_name}' 时发生未知错误: {e}")


# 定义输出文件名
# Define the output filename
output_file_name = 'output_list.json'

# 将合并后的数据写入新的JSON文件
# Write the merged data to a new JSON file
with open(output_file_name, 'w', encoding='utf-8') as f:
    # 使用json.dump写入数据
    # ensure_ascii=False 确保中文字符正确显示
    # indent=2 是为了让JSON文件格式化，更易于阅读
    # Use json.dump to write the data
    # ensure_ascii=False ensures that Chinese characters are displayed correctly
    # indent=2 is for formatting the JSON file to make it more readable
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"\n数据已成功合并到 '{output_file_name}' 文件中。")
print(f"总共合并了 {len(merged_data)} 条记录。")

