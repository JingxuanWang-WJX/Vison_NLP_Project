import io
import os
import re
import json
import tempfile
from typing import Optional
from PIL import Image
import gradio as gr

# 尝试导入 torch/transformers
try:
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration
    )

    try:
        from qwen_vl_utils import process_vision_info

        QWEN_VL_UTILS_AVAILABLE = True
    except ImportError:
        print("警告: qwen_vl_utils 不可用，将尝试替代方法")
        QWEN_VL_UTILS_AVAILABLE = False
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    QWEN_VL_UTILS_AVAILABLE = False

# ========== 配置区 ===========
# 模型路径配置
ORIGINAL_MODEL_PATH = ""  # 原始模型路径
FINETUNED_MODEL_PATH = ""  # 微调模型路径
GOOGLE_API_KEY = "AIzaSyBqrEvfCEbxMEADyy7A-dUpuEVwyHQAtsc"  # 你的Google API密钥
DEVICE = "cuda" if (torch and torch.cuda.is_available()) else "cpu"


# =============================

class BaseModelWrapper:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.ready = False

    def generate(self, pil_image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200):
        raise NotImplementedError("子类必须实现generate方法")


class LocalModelWrapper(BaseModelWrapper):
    def __init__(self, model_path: str, model_type: str, device_id: int = 0):  # 添加device_id参数
        super().__init__(model_type)
        self.model_path = model_path
        self.device_id = device_id  # 保存设备ID
        self.model = None
        self.processor = None
        if model_path:
            self._load()

    def _load(self):
        if not HF_AVAILABLE:
            print("transformers/torch not available: 不能自动加载 HF 模型")
            return

        try:
            print(f"尝试从 {self.model_path} 加载 processor/model 到 GPU {self.device_id}...")

            if not os.path.exists(self.model_path):
                print(f"模型路径不存在: {self.model_path}")
                return

            # 加载processor
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                print("✓ Processor 加载成功")
            except Exception as e:
                print(f"✗ Processor 加载失败: {e}")
                return

            # 加载模型 - 指定具体的GPU设备
            try:
                if DEVICE == "cuda":
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map={"": self.device_id}  # 使用指定的GPU设备
                    )
                else:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                print(f"✓ 模型加载成功到 GPU {self.device_id}")
            except Exception as e:
                print(f"✗ 模型加载失败: {e}")
                return

            if self.model is not None:
                self.model.eval()
                self.ready = True
                print("✓ 模型加载完成并设置为评估模式")
            else:
                print("✗ 模型对象为 None")

        except Exception as e:
            print(f"✗ 加载模型发生异常：{e}")
            import traceback
            traceback.print_exc()
            self.ready = False

    def generate(self, pil_image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200):
        if not self.ready:
            sample = (
                    f"[{self.model_type}模式] 模型未加载成功\n" +
                    f"提示词: {prompt}\n" +
                    ("已上传一张图片\n" if pil_image is not None else "未上传图片\n") +
                    "请检查模型路径和文件完整性。"
            )
            return sample

        try:
            # 使用 Qwen2.5-VL 官方推荐的消息格式
            if pil_image is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 处理视觉信息
            if QWEN_VL_UTILS_AVAILABLE:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [pil_image] if pil_image is not None else None
                video_inputs = None

            # 准备输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # 移动到指定的设备
            target_device = f"cuda:{self.device_id}" if DEVICE == "cuda" else DEVICE
            inputs = {k: v.to(target_device) for k, v in inputs.items() if hasattr(v, "to")}

            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor.tokenizer,
                                                                                  'eos_token_id') else None
                )

            # 解码
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return output_text[0] if output_text else ""

        except Exception as e:
            print(f"生成过程出错: {e}")
            import traceback
            traceback.print_exc()
            return f"生成失败: {str(e)}"


class GoogleAPIModelWrapper(BaseModelWrapper):
    def __init__(self, api_key: str):
        super().__init__("调用Gemini-2.5-Pro API创作诗句")
        self.api_key = api_key
        self.ready = True  # API模式总是就绪

    def generate(self, pil_image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200):
        try:
            # 导入Google GenAI库
            from google import genai

            # 初始化客户端
            client = genai.Client(api_key=self.api_key)

            # 创建临时目录保存文件
            with tempfile.TemporaryDirectory() as temp_dir:
                # 保存图片到临时文件
                image_path = os.path.join(temp_dir, "image.jpg")
                if pil_image:
                    pil_image.save(image_path, format="JPEG")
                else:
                    return "错误：调用Gemini-2.5-Pro API创作诗句需要上传图片"

                # 上传文件
                image_file = client.files.upload(file=image_path)
                prompt_file = client.files.upload(file="prompt.txt")
                instructions_file = client.files.upload(file="instructions.txt")

                # 调用API
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[prompt_file, image_file, instructions_file]
                )

            # 1. 从响应中获取原始文本内容
            raw_text = response.text

            # 2. 清理字符串，提取出纯净的JSON部分
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            json_string = ""
            if match:
                json_string = match.group(1)
            else:
                # 如果上面的正则匹配失败，尝试一个更宽松的模式
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                if start != -1 and end != -1:
                    json_string = raw_text[start:end + 1]

            # 检查是否成功提取了JSON字符串
            if not json_string:
                error_msg = "错误：无法在模型响应中找到有效的JSON内容。\n"
                error_msg += "原始响应文本：\n"
                error_msg += raw_text
                return error_msg

            try:
                # 3. 将清理后的JSON字符串解析为Python字典
                response_data = json.loads(json_string)

                # 4. 处理嵌套的JSON结构
                # 检查是否有 generated_content 嵌套结构
                if "generated_content" in response_data:
                    generated_data = response_data["generated_content"]
                else:
                    generated_data = response_data

                # 5. 构建输出格式 - 统一使用 poem_explanation 字段
                output = f"标题: {generated_data.get('poem_title', '无标题')}\n\n"
                output += f"诗歌内容:\n{generated_data.get('poem_content', '无内容')}\n\n"
                
                # 优先使用 poem_explanation，如果不存在则使用 poem_analysis
                explanation = generated_data.get('poem_explanation') or generated_data.get('poem_analysis', '无解析')
                output += f"解析:\n{explanation}"

                return output

            except json.JSONDecodeError as e:
                error_msg = f"错误：解析JSON时出错 - {e}\n"
                error_msg += "提取到的字符串是：\n"
                error_msg += json_string
                return error_msg

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"API调用失败: {str(e)}"


# ========== 初始化模型包装器 - 分配不同GPU ===========
api_wrapper = GoogleAPIModelWrapper(GOOGLE_API_KEY)
original_wrapper = LocalModelWrapper(ORIGINAL_MODEL_PATH, "Qwen2.5-VL-7B原始模型", device_id=0)  # GPU 0
finetuned_wrapper = LocalModelWrapper(FINETUNED_MODEL_PATH, "Qwen2.5-VL-7B LoRA微调模型", device_id=1)  # GPU 1


# ========== 诗歌输出格式化函数 ===========
def format_poem_output(result: str) -> str:
    """
    将诗歌生成结果格式化为美观的HTML
    """
    try:
        # 定义换行符变量，避免在 f-string 中使用反斜杠
        newline = '\n'
        br_tag = '<br>'

        # 尝试解析为JSON
        if result.startswith("{") and result.endswith("}"):
            response_data = json.loads(result)
            
            # 处理嵌套结构
            if "generated_content" in response_data:
                poem_data = response_data["generated_content"]
            else:
                poem_data = response_data
                
            poem_type = poem_data.get("poem_type", "诗歌")
            poem_title = poem_data.get("poem_title", "无题")
            poem_content = poem_data.get("poem_content", "")
            # 优先使用 poem_explanation，如果不存在则使用 poem_analysis
            poem_explanation = poem_data.get("poem_explanation") or poem_data.get("poem_analysis", "")

            # 创建美观的HTML输出
            html_output = f"""
            <div style="font-family: 'Ma Shan Zheng', 'Noto Sans SC', sans-serif; 
                        max-width: 800px; margin: 0 auto; padding: 20px;
                        background: linear-gradient(135deg, #f9f9f9 0%, #e8f4fc 100%);
                        border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">

                <!-- 诗歌类型徽章 -->
                <div style="text-align: center; margin-bottom: 20px;">
                    <span style="display: inline-block; background: linear-gradient(135deg, #3498db 0%, #1a5276 100%); 
                                color: white; padding: 8px 20px; 
                                border-radius: 30px; font-size: 1.1rem; font-weight: bold;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        {poem_type}
                    </span>
                </div>

                <!-- 诗歌标题 -->
                <h1 style="text-align: center; color: #2c3e50; font-size: 2.5rem; 
                            margin-bottom: 30px; border-bottom: 3px solid #f1c40f; 
                            padding-bottom: 15px; display: inline-block;
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                    {poem_title}
                </h1>

                <!-- 诗歌内容区域 -->
                <div style="background: url('https://www.transparenttextures.com/patterns/paper-fibers.png'), 
                            linear-gradient(to bottom, #fcf5e9, #f9f0e1);
                            padding: 40px 30px; 
                            border-radius: 10px; text-align: center; 
                            margin-bottom: 30px; border: 1px solid #e8d8b6; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                            position: relative;">
                    <div style="position: absolute; top: 10px; left: 10px; 
                                font-family: 'Ma Shan Zheng', cursive; color: #5c3b1a;
                                font-size: 1.2rem; opacity: 0.7;">
                        诗意视觉
                    </div>
                    <div style="position: absolute; bottom: 10px; right: 10px; 
                                font-family: 'Ma Shan Zheng', cursive; color: #5c3b1a;
                                font-size: 1.2rem; opacity: 0.7;">
                        AI创作
                    </div>
                    <div style="font-size: 1.8rem; line-height: 2.2; 
                                color: #5c3b1a; letter-spacing: 1px; 
                                font-family: 'Ma Shan Zheng', cursive;">
                        {poem_content.replace(newline, br_tag)}
                    </div>
                </div>

                <!-- 诗歌解析 -->
                <div style="background: linear-gradient(to right, #f8f9fa, #e3f2fd); 
                            padding: 25px; 
                            border-radius: 10px; border-left: 5px solid #3498db;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                    <h2 style="color: #2c3e50; font-size: 1.5rem; 
                            margin-top: 0; margin-bottom: 20px; 
                            display: flex; align-items: center;">
                        <span style="background: linear-gradient(135deg, #3498db 0%, #1a5276 100%); 
                                    color: white; width: 35px; height: 35px; border-radius: 50%; 
                                    display: inline-flex; align-items: center; 
                                    justify-content: center; margin-right: 10px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                            析
                        </span>
                        诗歌解析
                    </h2>

                    <div style="font-size: 1.1rem; line-height: 1.8; color: #34495e;
                                background: rgba(255,255,255,0.7); padding: 15px;
                                border-radius: 8px;">
                        {poem_explanation}
                    </div>
                </div>

                <!-- 装饰元素 -->
                <div style="text-align: center; margin-top: 30px; 
                            color: #7f8c8d; font-size: 1rem;">
                    <div style="font-family: 'ZCOOL XiaoWei', serif;">
                        ✦ 诗意视觉 · AI诗歌创作 ✦
                    </div>
                </div>
            </div>
            """
            return html_output

        # 如果不是JSON格式，尝试提取标题和内容
        title_match = re.search(r"标题: (.+)", result)
        content_match = re.search(r"诗歌内容:\n([\s\S]+?)\n\n解析:", result)
        explanation_match = re.search(r"解析:\n([\s\S]+)", result)

        if title_match and content_match and explanation_match:
            poem_title = title_match.group(1)
            poem_content = content_match.group(1)
            poem_explanation = explanation_match.group(1)

            # 创建美观的HTML输出
            return f"""
            <div style="font-family: 'Ma Shan Zheng', 'Noto Sans SC', sans-serif; 
                        max-width: 800px; margin: 0 auto; padding: 20px;
                        background: linear-gradient(135deg, #f9f9f9 0%, #e8f4fc 100%);
                        border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">

                <!-- 诗歌标题 -->
                <h1 style="text-align: center; color: #2c3e50; font-size: 2.5rem; 
                            margin-bottom: 30px; border-bottom: 3px solid #f1c40f; 
                            padding-bottom: 15px; display: inline-block;
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                    {poem_title}
                </h1>

                <!-- 诗歌内容区域 -->
                <div style="background: url('https://www.transparenttextures.com/patterns/paper-fibers.png'), 
                            linear-gradient(to bottom, #fcf5e9, #f9f0e1);
                            padding: 40px 30px; 
                            border-radius: 10px; text-align: center; 
                            margin-bottom: 30px; border: 1px solid #e8d8b6; 
                            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                            position: relative;">
                    <div style="position: absolute; top: 10px; left: 10px; 
                                font-family: 'Ma Shan Zheng', cursive; color: #5c3b1a;
                                font-size: 1.2rem; opacity: 0.7;">
                        诗意视觉
                    </div>
                    <div style="position: absolute; bottom: 10px; right: 10px; 
                                font-family: 'Ma Shan Zheng', cursive; color: #5c3b1a;
                                font-size: 1.2rem; opacity: 0.7;">
                        AI创作
                    </div>
                    <div style="font-size: 1.8rem; line-height: 2.2; 
                                color: #5c3b1a; letter-spacing: 1px; 
                                font-family: 'Ma Shan Zheng', cursive;">
                        {poem_content.replace(newline, br_tag)}
                    </div>
                </div>

                <!-- 诗歌解析 -->
                <div style="background: linear-gradient(to right, #f8f9fa, #e3f2fd); 
                            padding: 25px; 
                            border-radius: 10px; border-left: 5px solid #3498db;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                    <h2 style="color: #2c3e50; font-size: 1.5rem; 
                            margin-top: 0; margin-bottom: 20px; 
                            display: flex; align-items: center;">
                        <span style="background: linear-gradient(135deg, #3498db 0%, #1a5276 100%); 
                                    color: white; width: 35px; height: 35px; border-radius: 50%; 
                                    display: inline-flex; align-items: center; 
                                    justify-content: center; margin-right: 10px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                            析
                        </span>
                        诗歌解析
                    </h2>

                    <div style="font-size: 1.1rem; line-height: 1.8; color: #34495e;
                                background: rgba(255,255,255,0.7); padding: 15px;
                                border-radius: 8px;">
                        {poem_explanation}
                    </div>
                </div>

                <!-- 装饰元素 -->
                <div style="text-align: center; margin-top: 30px; 
                            color: #7f8c8d; font-size: 1rem;">
                    <div style="font-family: 'ZCOOL XiaoWei', serif;">
                        ✦ 诗意视觉 · AI诗歌创作 ✦
                    </div>
                </div>
            </div>
            """

    except Exception as e:
        print(f"格式化诗歌输出时出错: {e}")

    # 如果无法解析，返回原始结果（用<pre>标签保留格式）
    return f"<pre style='white-space: pre-wrap; background: #f8f9fa; padding: 20px; border-radius: 10px;'>{result}</pre>"


# ========== Gradio 回调函数 ===========
def predict(image: Optional[Image.Image], prompt: str, max_new_tokens: int, mode: str):
    """根据选择的模式调用相应的生成器"""
    if image is not None and not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    result = ""
    if mode == "调用Gemini-2.5-Pro API创作诗句":
        result = api_wrapper.generate(image, prompt, max_new_tokens)
    elif mode == "Qwen2.5-VL-7B原始模型":
        result = original_wrapper.generate(image, prompt, max_new_tokens)
    elif mode == "Qwen2.5-VL-7B LoRA微调模型":
        result = finetuned_wrapper.generate(image, prompt, max_new_tokens)
    else:
        result = "未知模式，请选择正确的生成模式"

    # 格式化输出结果
    return format_poem_output(result)


def load_demo_example():
    return None, "请根据图片写一首现代汉语诗，情感细腻、意象丰富："


# ========== 创建多页面 Gradio 应用 ===========
# 自定义CSS样式
custom_css = """
/* 整体风格 */
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Noto Sans SC', sans-serif;
}

/* 主标题 */
h1 {
    font-family: 'Ma Shan Zheng', cursive;
    font-size: 2.8rem !important;
    color: #2c3e50;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}

/* 按钮样式 */
button {
    font-family: 'ZCOOL XiaoWei', serif !important;
    font-size: 1.2rem !important;
    border-radius: 50px !important;
    padding: 15px 30px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    border: none !important;
}

button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
}

/* 输入输出区域 */
.gr-box {
    border-radius: 15px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08) !important;
    border: 1px solid #e0e0e0 !important;
    background: rgba(255,255,255,0.9) !important;
}

/* 标签样式 */
label {
    font-family: 'ZCOOL XiaoWei', serif !important;
    font-size: 1.2rem !important;
    color: #2c3e50 !important;
    font-weight: 600 !important;
}

/* 页面切换动画 */
[class^="page-"] {
    animation: fadeIn 0.6s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 返回按钮 */
#back_btn {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%) !important;
    color: white !important;
    margin-bottom: 20px;
}

/* 生成按钮 */
#generate_btn {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.3rem !important;
}

/* 示例按钮 */
#example_btn {
    background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%) !important;
    color: #2c3e50 !important;
}

/* 模式按钮 */
#mode_btn {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
    color: white !important;
    margin: 10px;
}

/* 诗歌输出区域 */
#poem_output {
    font-family: 'Ma Shan Zheng', cursive !important;
    font-size: 1.4rem !important;
    line-height: 1.8;
    color: #2c3e50;
    background: linear-gradient(to bottom, #ffffff, #f8f9fa) !important;
    border-left: 5px solid #3498db !important;
    padding: 20px !important;
    border-radius: 15px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
}

/* 页脚 */
footer {
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    color: #7f8c8d;
    font-family: 'ZCOOL XiaoWei', serif;
    border-top: 1px solid #ecf0f1;
}

/* 输入区域 */
.gr-input {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    border: 1px solid #e0e0e0 !important;
    padding: 20px !important;
}

/* 输出标题 */
.output-title {
    font-family: 'Ma Shan Zheng', cursive !important;
    font-size: 1.8rem !important;
    color: #2c3e50 !important;
    margin-bottom: 15px !important;
    text-align: center;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}
"""

with gr.Blocks(title="多模式诗歌生成系统", css=custom_css) as demo:
    # 添加自定义字体
    demo.head = """
    <link href="https://fonts.googleapis.com/css2?family=Ma+Shan+Zheng&family=ZCOOL+XiaoWei&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    """

    # 状态变量，跟踪当前页面
    current_page = gr.State(value="main")

    # 主页面 - 模式选择
    with gr.Column(visible=True, elem_id="main") as main_page:
        gr.Markdown("""
        <div style="text-align:center; background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); 
                    padding: 30px; border-radius: 15px; margin-bottom: 30px; color: white;">
            <h1 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">✨ 诗意视觉 · 多模式诗歌生成系统 ✨</h1>
            <p style="font-size:1.2rem; max-width:800px; margin:0 auto;">
                融合视觉与语言的艺术创作平台，让AI为您创作独特的诗歌
            </p>
        </div>
        """)

        gr.Markdown("### 🎯 请选择生成模式", elem_classes="page-title")

        with gr.Row():
            api_btn = gr.Button("🌐 调用Gemini-2.5-Pro API创作诗句", variant="primary", elem_id="mode_btn")
            original_btn = gr.Button("🧠 Qwen2.5-VL-7B原始模型", variant="primary", elem_id="mode_btn")
            finetuned_btn = gr.Button("🎨 Qwen2.5-VL-7B LoRA微调模型", variant="primary", elem_id="mode_btn")

        gr.Markdown("### 📚 模式说明")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                    <h3 style="color: #3498db; font-family: 'Ma Shan Zheng', cursive;">🌐 调用Gemini-2.5-Pro API创作诗句</h3>
                    <ul>
                        <li>通过调用Gemini-2.5-Pro API创作诗句</li>
                        <li>快速响应，无需本地计算资源</li>
                        <li>适合快速原型和演示</li>
                    </ul>
                </div>
                """)
            with gr.Column():
                gr.Markdown("""
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                    <h3 style="color: #e74c3c; font-family: 'Ma Shan Zheng', cursive;">🧠 Qwen2.5-VL-7B原始模型</h3>
                    <ul>
                        <li>使用原始Qwen2.5-VL-7B原始模型生成诗歌</li>
                        <li>强大的多模态理解能力</li>
                        <li>通用性强，适合多种创作场景</li>
                    </ul>
                </div>
                """)
            with gr.Column():
                gr.Markdown("""
                <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 15px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
                    <h3 style="color: #9b59b6; font-family: 'Ma Shan Zheng', cursive;">🎨 Qwen2.5-VL-7B LoRA微调模型</h3>
                    <ul>
                        <li>使用经过高质量诗歌数据集LoRA微调的Qwen2.5-VL-7B模型</li>
                        <li>专为诗歌创作优化，专注于五言绝句、七言绝句、五言律诗、七言律诗、词的创作</li>
                        <li>生成更具文学性和艺术性的作品</li>
                    </ul>
                </div>
                """)

        gr.Markdown("---")
        gr.Markdown("""
        <div style="text-align:center; padding:20px; background: linear-gradient(135deg, #f9f9f9 0%, #e8f4fc 100%); 
                    border-radius:15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            <p style="font-family:'ZCOOL XiaoWei', serif; font-size:1.2rem;">
            上传一张图片，输入创作提示，选择您喜欢的模式，即可生成独特的诗歌作品
            </p>
        </div>
        """)

        # 页脚
        gr.Markdown("""
        <footer>
            <p>Qwen2.5-VL-7B 多模式诗歌生成系统 © 2024</p>
            <p>技术支持：通义千问 · 多模态大模型</p>
        </footer>
        """)

    # API调用模式页面
    with gr.Column(visible=False, elem_id="api_page", elem_classes="page-api") as api_page:
        gr.Markdown("# 🌐 调用Gemini-2.5-Pro API创作诗句")
        back_btn_api = gr.Button("← 返回主菜单", elem_id="back_btn")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    input_image_api = gr.Image(type="pil", label="🖼️ 上传图片（点击或拖放）")
                    prompt_box_api = gr.Textbox(lines=3, value="请为这张图片创作一首诗词并加以解读。",
                                                label="📝 创作提示")
                    max_tokens_api = gr.Slider(minimum=2048, maximum=4096, step=1, value=200, label="生成长度 (tokens)")

                with gr.Row():
                    run_btn_api = gr.Button("✨ 生成诗句", variant="primary", elem_id="generate_btn")
                    example_btn_api = gr.Button("📋 加载示例提示", elem_id="example_btn")

            with gr.Column(scale=1):
                gr.Markdown("### 📜 生成的诗句", elem_classes="output-title")
                output_box_api = gr.HTML(elem_id="poem_output")

    # Qwen2.5-VL-7B原始模型页面
    with gr.Column(visible=False, elem_id="original_page", elem_classes="page-original") as original_page:
        gr.Markdown("# 🧠 Qwen2.5-VL-7B原始模型")
        back_btn_original = gr.Button("← 返回主菜单", elem_id="back_btn")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    input_image_original = gr.Image(type="pil", label="🖼️ 上传图片（点击或拖放）")
                    prompt_box_original = gr.Textbox(lines=3, value="请为这张图片创作一首诗词并加以解读。",
                                                     label="📝 创作提示")
                    max_tokens_original = gr.Slider(minimum=2048, maximum=4096, step=1, value=200,
                                                    label="生成长度 (tokens)")

                with gr.Row():
                    run_btn_original = gr.Button("✨ 生成诗句", variant="primary", elem_id="generate_btn")
                    example_btn_original = gr.Button("📋 加载示例提示", elem_id="example_btn")

            with gr.Column(scale=1):
                gr.Markdown("### 📜 生成的诗句", elem_classes="output-title")
                output_box_original = gr.HTML(elem_id="poem_output")

    # Qwen2.5-VL-7B LoRA微调模型页面
    with gr.Column(visible=False, elem_id="finetuned_page", elem_classes="page-finetuned") as finetuned_page:
        gr.Markdown("# 🎨 Qwen2.5-VL-7B LoRA微调模型")
        back_btn_finetuned = gr.Button("← 返回主菜单", elem_id="back_btn")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    input_image_finetuned = gr.Image(type="pil", label="🖼️ 上传图片（点击或拖放）")
                    prompt_box_finetuned = gr.Textbox(lines=3, value="请为这张图片创作一首诗词并加以解读。",
                                                      label="📝 创作提示")
                    max_tokens_finetuned = gr.Slider(minimum=2048, maximum=4096, step=1, value=200,
                                                     label="生成长度 (tokens)")

                with gr.Row():
                    run_btn_finetuned = gr.Button("✨ 生成诗句", variant="primary", elem_id="generate_btn")
                    example_btn_finetuned = gr.Button("📋 加载示例提示", elem_id="example_btn")

            with gr.Column(scale=1):
                gr.Markdown("### 📜 生成的诗句", elem_classes="output-title")
                output_box_finetuned = gr.HTML(elem_id="poem_output")


    # ========== 页面导航逻辑 ===========
    def navigate_to_page(page):
        """导航到指定页面"""
        pages = {
            "main": [True, False, False, False],
            "api": [False, True, False, False],
            "original": [False, False, True, False],
            "finetuned": [False, False, False, True]
        }
        return [
            gr.update(visible=pages[page][0]),  # main_page
            gr.update(visible=pages[page][1]),  # api_page
            gr.update(visible=pages[page][2]),  # original_page
            gr.update(visible=pages[page][3]),  # finetuned_page
            page  # 更新当前页面状态
        ]


    # 主页面按钮事件
    api_btn.click(lambda: navigate_to_page("api"),
                  outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    original_btn.click(lambda: navigate_to_page("original"),
                       outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    finetuned_btn.click(lambda: navigate_to_page("finetuned"),
                        outputs=[main_page, api_page, original_page, finetuned_page, current_page])

    # 返回按钮事件
    back_btn_api.click(lambda: navigate_to_page("main"),
                       outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    back_btn_original.click(lambda: navigate_to_page("main"),
                            outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    back_btn_finetuned.click(lambda: navigate_to_page("main"),
                             outputs=[main_page, api_page, original_page, finetuned_page, current_page])

    # 示例按钮事件
    example_btn_api.click(load_demo_example, outputs=[input_image_api, prompt_box_api])
    example_btn_original.click(load_demo_example, outputs=[input_image_original, prompt_box_original])
    example_btn_finetuned.click(load_demo_example, outputs=[input_image_finetuned, prompt_box_finetuned])

    # 生成按钮事件
    run_btn_api.click(
        fn=predict,
        inputs=[input_image_api, prompt_box_api, max_tokens_api, gr.State("调用Gemini-2.5-Pro API创作诗句")],
        outputs=[output_box_api]
    )
    run_btn_original.click(
        fn=predict,
        inputs=[input_image_original, prompt_box_original, max_tokens_original, gr.State("Qwen2.5-VL-7B原始模型")],
        outputs=[output_box_original]
    )
    run_btn_finetuned.click(
        fn=predict,
        inputs=[input_image_finetuned, prompt_box_finetuned, max_tokens_finetuned,
                gr.State("Qwen2.5-VL-7B LoRA微调模型")],
        outputs=[output_box_finetuned]
    )

if __name__ == '__main__':
    demo.launch(share=True)