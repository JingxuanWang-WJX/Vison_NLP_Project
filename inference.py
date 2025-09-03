"""
gradio_qwen_vl_demo.py

可运行的 Gradio 演示脚本，用于验证你导出的 Qwen2.5-VL-3B（或其他多模态模型）是否能根据图片+提示生成诗句。

使用说明：
1. 修改下面的 MODEL_PATH 为你导出的模型目录或 Hugging Face 模型标识（如果用 HF）。
2. 如果模型加载方式与示例不兼容，请在 load_model() 中替换为你自己的加载代码。
3. 运行：
    pip install -r requirements.txt
    python gradio_qwen_vl_demo.py

requirements.txt（示例）:
    gradio
    pillow
    torch
    transformers

这个脚本做了两件事：
- 为常见的 HF 风格多模态模型提供一个"尝试加载"的实现（AutoProcessor + AutoModel），并尽力把图片传入 model.generate()。
- 如果你不想马上加载模型，脚本也提供了一个回退的简单 "fake" 生成器，方便 UI 测试。

注意：不同模型接受图片的字段名可能不同（pixel_values / images / image_tensors / vision_outputs 等）。本脚本在调用生成时尝试了多种常见字段名，但无法覆盖所有自定义导出格式，必要时请根据你的导出 API 修改 generate_with_model()。

"""

import io
import os
from typing import Optional

from PIL import Image
import gradio as gr

# 尝试导入 torch/transformers（如果没有请先 pip install）
try:
    import torch
    from transformers import (
        AutoProcessor, 
        AutoTokenizer, 
        Qwen2_5_VLForConditionalGeneration,
        AutoModel,
        AutoModelForCausalLM
    )
    # 添加 qwen_vl_utils 导入
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


# ========== 配置区（请按需修改） ===========
# 本地或 HF 模型路径（可留空以使用回退生成器）
MODEL_PATH = "/mnt/e/code/Pycharmcode/VLM/Finetune/Finetune"  # e.g. "/path/to/your/exported/qwen2.5-vl-3b" or "username/qwen2.5-vl-3b"
DEVICE = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
# ===========================================


class ModelWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.ready = False
        if model_path:
            self._load()

    def _load(self):
        if not HF_AVAILABLE:
            print("transformers/torch not available: 不能自动加载 HF 模型，请安装依赖或提供自定义加载代码。")
            return

        try:
            print(f"尝试从 {self.model_path} 加载 processor/tokenizer/model...")
            
            # 首先检查模型路径是否存在
            if not os.path.exists(self.model_path):
                print(f"模型路径不存在: {self.model_path}")
                return
            
            # 检查必要文件是否存在
            required_files = ['config.json']
            for file in required_files:
                if not os.path.exists(os.path.join(self.model_path, file)):
                    print(f"缺少必要文件: {file}")
                    return
            
            # 加载processor（包含图像处理器和tokenizer）
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
                print("✓ Processor 加载成功")
            except Exception as e:
                print(f"✗ Processor 加载失败: {e}")
                return

            # 使用更通用的方式加载模型
            try:
                # 修改模型加载配置，避免多GPU分布
                if DEVICE == "cuda":
                    # 如果是单GPU，直接指定设备
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        # 不使用 device_map="auto"，改为手动指定
                        device_map={"": 0}  # 将所有模块放在GPU 0上
                    )
                else:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                print("✓ AutoModel 加载成功")
            except Exception as e:
                print(f"✗ AutoModel 加载失败: {e}")
                return

            if self.model is not None:
                # 由于使用了device_map，不需要再手动.to(device)
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
                "[回退生成器] 未加载模型，下面是用于测试的伪输出。\n" +
                f"提示词: {prompt}\n" +
                ("已上传一张图片，大小: %dx%d\n" % pil_image.size if pil_image is not None else "未上传图片\n") +
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
                            {
                                "type": "image",
                                "image": pil_image,  # 直接传入 PIL 图像
                            },
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
                # 如果没有 qwen_vl_utils，尝试手动提取图像
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
            
            # 移动到设备 - 确保使用与模型相同的设备
            target_device = "cuda:0" if DEVICE == "cuda" else DEVICE
            inputs = {k: v.to(target_device) for k, v in inputs.items() if hasattr(v, "to")}
            
            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor.tokenizer, 'eos_token_id') else None
                )
            
            # 解码（只返回新生成的部分）
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


# ========== 初始化模型包装器 ===========
model_wrapper = ModelWrapper(MODEL_PATH)


# ========== Gradio 回调 ===========

def predict(image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200):
    """Gradio 回调：接受 PIL.Image 和 prompt 返回生成文本"""
    if image is not None and not isinstance(image, Image.Image):
        # gradio 有时给出 numpy 数组
        image = Image.fromarray(image)
    output = model_wrapper.generate(image, prompt, max_new_tokens=max_new_tokens)
    return output


def load_demo_example():
    # 预置示例图片（如果你在本地有图片，可以更改路径）
    return None, "请根据图片写一首现代汉语诗，情感细腻、意象丰富："


# ========== Gradio 界面 ===========
with gr.Blocks(title="Qwen2.5-VL-3B 图文生成诗句 Demo") as demo:
    gr.Markdown("## Qwen 图文生成 Demo\n上传图片 + 输入提示词，模型将尝试根据图片生成诗句。")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="上传图片（点击或拖放）")
            prompt_box = gr.Textbox(lines=3, placeholder="在此输入提示词（例如：请根据图片写一首现代诗）", label="Prompt")
            max_tokens = gr.Slider(minimum=16, maximum=512, step=1, value=200, label="生成最大 tokens")
            run_btn = gr.Button("生成诗句")
            example_btn = gr.Button("加载示例提示")
        with gr.Column(scale=1):
            output_box = gr.Textbox(label="生成的诗句", lines=12)

    run_btn.click(fn=predict, inputs=[input_image, prompt_box, max_tokens], outputs=[output_box])
    example_btn.click(fn=load_demo_example, inputs=[], outputs=[input_image, prompt_box])

    gr.Markdown("---\n注意：如果你的模型在远端或需要特殊的请求格式，请在本脚本中替换 ModelWrapper._load 和 ModelWrapper.generate 的实现。")


if __name__ == '__main__':
    demo.launch(share=False)
