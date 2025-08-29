import json
import os


def convert_to_finetune_format(input_file, output_file):
    """
    将包含多个数据条目的JSON列表文件，转换为大模型微调所需的数据格式，
    并将其追加到输出文件的列表中。

    参数:
    input_file (str): 输入的JSON文件名 (例如 'output_list_v1.json')。
    output_file (str): 输出的JSON文件名 (例如 'finetune_data.json')。
    """
    try:
        # --- 步骤 1: 读取包含列表的原始JSON文件 ---
        # 使用 'with' 语句可以确保文件被正确关闭
        # 指定 encoding='utf-8' 来正确处理中文字符
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data_list = json.load(f)

        # 检查输入数据是否为列表
        if not isinstance(source_data_list, list):
            print(f"错误: '{input_file}' 的内容不是一个列表。脚本将退出。")
            return

        # --- 步骤 2: 读取现有数据（如果文件存在）并准备追加 ---
        # 检查输出文件是否存在且不为空
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    # 尝试加载现有数据
                    existing_data = json.load(f)
                # 确保它是一个列表
                if not isinstance(existing_data, list):
                    print(f"警告: '{output_file}' 已存在但不是列表格式。将用新列表覆盖。")
                    existing_data = []
            except json.JSONDecodeError:
                print(f"警告: 无法解析 '{output_file}' 的内容。将用新列表覆盖。")
                existing_data = []
        else:
            # 如果文件不存在或是空的，就从一个空列表开始
            existing_data = []

        # --- 步骤 3: 遍历输入列表中的每一个数据条目 ---
        for item in source_data_list:
            # --- 步骤 3a: 提取需要的数据 ---
            # 从每个条目中获取图片路径和生成的内容
            relative_path = item.get("image_path")
            assistant_content_obj = item.get("generated_content", {})

            # 如果生成失败 (create_condition 为 false)，则跳过该条目
            if not assistant_content_obj.get("create_condition", True):
                print(f"跳过条目 (image: {relative_path})，因为生成条件为 false。")
                continue

            # 检查关键信息是否存在
            if not relative_path:
                print(f"警告: 在 '{input_file}' 的一个条目中找不到 'image_path'。")
                continue

            # --- 新增步骤: 将 image_path 转换为指定的绝对路径格式 ---
            # 定义基础路径
            base_path = "/home/wjx/VLM/data"
            # 将 Windows 风格的路径分隔符 '\\' 替换为 '/'
            normalized_path = relative_path.replace('\\', '/')
            # 拼接成完整的绝对路径
            image_path = f"{base_path}/{normalized_path}"

            # --- 步骤 3b: 移除不需要的键 ---
            assistant_content_obj.pop("create_condition", None)

            # --- 步骤 3c: 构建用户和助手的内容 ---
            # 用户内容是图片占位符和一个通用的提示
            user_content = "<image>请为这张图片创作一首诗词并加以解读。"
            # 助手内容是结构化数据的JSON字符串
            assistant_response_str = json.dumps(assistant_content_obj, ensure_ascii=False)

            # --- 步骤 3d: 构建单个数据条目的JSON对象 ---
            new_entry = {
                "messages": [
                    {
                        "content": user_content,
                        "role": "user"
                    },
                    {
                        "content": assistant_response_str,
                        "role": "assistant"
                    }
                ],
                "images": [
                    image_path  # 使用新生成的绝对路径
                ]
            }
            # 将新的数据条目追加到列表中
            existing_data.append(new_entry)

        # --- 步骤 4: 将包含所有条目的列表写入新的JSON文件 ---
        # ensure_ascii=False 确保中文字符在文件中能正常显示
        # indent=4 让输出的JSON文件格式更美观，易于阅读
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"数据已成功从 '{input_file}' 转换并保存到 '{output_file}'")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_file}'。请检查文件名和路径是否正确。")
    except json.JSONDecodeError:
        print(f"错误：无法解析 '{input_file}'。请检查文件内容是否为有效的JSON格式。")
    except Exception as e:
        print(f"发生了一个未知错误: {e}")


# --- 如何使用 ---
if __name__ == "__main__":
    # 假设你的源数据文件名为 'output_list_v1.json'
    # 这个文件需要是一个列表，其中每个对象都包含 "image_path" 和 "generated_content"
    input_json_file = 'output_list.json'

    # 设置你希望保存微调数据的文件名
    output_json_file = 'finetune_data.json'

    convert_to_finetune_format(input_json_file, output_json_file)
