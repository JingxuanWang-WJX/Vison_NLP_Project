import os
from google import genai
import json
import re

# 配置你的API密钥
# 请将 "YOUR_API_KEY" 替换为你的真实密钥
client = genai.Client(api_key="AIzaSyBqrEvfCEbxMEADyy7A-dUpuEVwyHQAtsc")

# 新增：定义要处理的文件夹和最终输出文件
IMAGE_DIRECTORIES = ["Train", "Val"]
OUTPUT_FILE = "output_list.json"

# 存储所有图片处理结果的列表
all_results = []

print("正在上传公用的 prompt.txt 和 instructions.txt 文件...")
# 将 prompt 和 instructions 文件预先上传一次，避免在循环中重复上传
prompt_file = client.files.upload(file="prompt.txt")
instructions_file = client.files.upload(file="instructions.txt")
print("上传完成。")

# 初始化计数器
i = 0

# 遍历指定的图片文件夹
for directory in IMAGE_DIRECTORIES:
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"警告：目录 '{directory}' 不存在，已跳过。")
        continue

    print(f"\n--- 开始处理文件夹: {directory} ---")
    image_files = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
    image_files = sorted(image_files)  # 按文件名排序，确保处理顺序一致

    for filename in image_files:
        image_path = os.path.join(directory, filename)
        print(f"正在处理图片: {image_path}")

        try:
            # --- 以下为你的原始代码逻辑 ---
            image_file = client.files.upload(file=image_path)

            response = client.models.generate_content(
                model="gemini-2.5-pro", contents=[prompt_file, image_file, instructions_file]
            )

            # 1. 从响应中获取原始文本内容
            raw_text = response.text

            # 2. 清理字符串，提取出纯净的JSON部分
            # 使用正则表达式查找被 ```json 和 ``` 包裹的内容
            # re.DOTALL 标志让 . 可以匹配包括换行符在内的任意字符
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            json_string = ""
            if match:
                json_string = match.group(1)
            else:
                # 如果上面的正则匹配失败，尝试一个更宽松的模式，直接找第一个和最后一个花括号
                # 这可以应对模型输出不完全符合 ```json ... ``` 格式的情况
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                if start != -1 and end != -1:
                    json_string = raw_text[start:end+1]

            # 检查是否成功提取了JSON字符串
            if not json_string:
                print(f"错误：无法在 {image_path} 的响应中找到有效的JSON内容。")
                # 清理本次上传的图片文件
                client.files.delete(name=image_file.name)
                continue # 继续处理下一张图片

            # 3. 将清理后的JSON字符串解析为Python字典
            generated_data = json.loads(json_string)

            # 4. 从响应的元数据中获取token总数
            total_token_count = response.usage_metadata.total_token_count

            # 5. 构建一个新的字典来存储所需信息
            final_result = {
                "generated_content": generated_data,
                "total_token_count": total_token_count,
                "image_path": image_path
            }

            # 将处理结果存入列表
            all_results.append(final_result)
            print(f"  成功处理并记录: {image_path}")

            # 清理本次上传的图片文件
            client.files.delete(name=image_file.name)

            # 计数器加一
            i += 1

            if i % 10 == 0:
                with open(f"output_list_{i}.json", "w", encoding="utf-8") as f:
                    # 将整个列表写入文件，格式为JSON
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"前{i}张图片处理完成已成功保存！")

        except json.JSONDecodeError as e:
            print(f"错误：解析来自 {image_path} 的JSON时出错 - {e}")
            print("  提取到的字符串是：", json_string)
            if 'image_file' in locals():
                client.files.delete(name=image_file.name)
            continue
        except Exception as e:
            print(f"处理 {image_path} 时发生未知错误: {e}")
            if 'image_file' in locals():
                client.files.delete(name=image_file.name)
            continue

# 清理公用的 prompt 和 instructions 文件
print("\n--- 处理完毕，正在清理公用文件 ---")
client.files.delete(name=prompt_file.name)
client.files.delete(name=instructions_file.name)

# 将所有结果一次性写入到最终的列表文件中
print(f"\n正在将所有结果写入到文件 {OUTPUT_FILE}...")
try:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 将整个列表写入文件，格式为JSON
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("所有结果已成功保存！")
except Exception as e:
    print(f"错误：写入文件失败。{e}")
