import json
import os


def delete_images_from_json(json_file_path):
    """
    读取JSON文件，获取图片路径并删除相应的图片。

    参数:
    json_file_path (str): 包含图片路径的JSON文件的路径。
    """
    # 检查JSON文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：找不到JSON文件 '{json_file_path}'")
        return

    try:
        # 打开并加载JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 '{json_file_path}'。请检查文件格式。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误：{e}")
        return

    # 检查JSON数据是否是列表
    if not isinstance(data, list):
        print("错误：JSON文件的顶层结构应该是一个列表 [...]。")
        return

    print("开始处理图片删除任务...")
    deleted_count = 0
    not_found_count = 0

    # 遍历JSON数据中的每个项目
    for item in data:
        # 检查项目是否是字典并且包含'image_path'键
        if isinstance(item, dict) and 'image_path' in item:
            image_path = item['image_path']

            # 规范化路径，替换反斜杠为适合当前操作系统的分隔符
            # 这使得脚本在Windows, Linux, macOS上都能更好地工作
            normalized_path = os.path.normpath(image_path)

            # 检查文件是否存在
            if os.path.exists(normalized_path):
                try:
                    # 如果文件存在，则删除它
                    os.remove(normalized_path)
                    print(f"已删除图片: {normalized_path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"删除文件时出错 '{normalized_path}': {e}")
            else:
                # 如果文件不存在，则打印提示信息
                print(f"图片未找到，跳过: {normalized_path}")
                not_found_count += 1
        else:
            print("警告：JSON中的一个项目格式不正确，已跳过。")

    print("\n任务完成。")
    print(f"总计：成功删除 {deleted_count} 张图片，{not_found_count} 张图片未找到。")


# --- 使用说明 ---
# 1. 将此脚本保存为 .py 文件（例如 delete_script.py）。
# 2. 将此脚本与您的 output_list_470_tem1.json 文件放在同一个文件夹中。
# 3. 确保JSON文件中 'image_path' 的路径（例如 "Train\\0.jpg"）是相对于此脚本的正确路径。
#    这意味着您应该有一个名为 "Train" 的子文件夹，并且图片在其中。
# 4. 在终端或命令行中运行此脚本： python delete_script.py

if __name__ == '__main__':
    # JSON文件的名称
    json_filename = 'output_list_tem8.json'

    # 执行删除函数
    delete_images_from_json(json_filename)
