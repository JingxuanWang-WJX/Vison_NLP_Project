from google import genai
import json
import re

client = genai.Client(api_key="AIzaSyBqrEvfCEbxMEADyy7A-dUpuEVwyHQAtsc")

image_file = client.files.upload(file="Train/10.jpg")

prompt_file = client.files.upload(file="prompt.txt")

instructions_file = client.files.upload(file="instructions.txt")

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
    print("错误：无法在模型响应中找到有效的JSON内容。")
    print("原始响应文本：")
    print(raw_text)
else:
    try:
        # 3. 将清理后的JSON字符串解析为Python字典
        generated_data = json.loads(json_string)

        # 4. 从响应的元数据中获取token总数
        total_token_count = response.usage_metadata.total_token_count

        # 5. 构建一个新的字典来存储所需信息
        final_result = {
            "generated_content": generated_data,
            "total_token_count": total_token_count
        }

        # 6. 将最终的字典转换为格式化的JSON字符串并打印
        # indent=2 用于美化输出，ensure_ascii=False 确保中文字符正确显示
        print(json.dumps(final_result, indent=2, ensure_ascii=False))

        # 7. 将最终结果写入文件
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)

    except json.JSONDecodeError as e:
        print(f"错误：解析JSON时出错 - {e}")
        print("提取到的字符串是：")
        print(json_string)
