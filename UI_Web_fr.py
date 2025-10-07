import io
import os
import re
import json
import tempfile
from typing import Optional
from PIL import Image
import gradio as gr
import textwrap
import html

# Load torch / transformers if available
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
        print("Warning: qwen_vl_utils not available, fallback methods will be used.")
        QWEN_VL_UTILS_AVAILABLE = False
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    QWEN_VL_UTILS_AVAILABLE = False

# ================= Configuration =================
# Model paths
ORIGINAL_MODEL_PATH = "/home/data2/Qwen2.5-VL-7B-Instruct/"
FINETUNED_MODEL_PATH = "/home/wjx/VLM/Finetune/"
# Use environment variable for safety (export GOOGLE_API_KEY=xxxx)
GOOGLE_API_KEY = "AIzaSyCns_L2WFkV9kT6B5F1h8rDE2ULrz_NUF8"  # 你的Google API密钥
DEVICE = "cuda" if ('torch' in globals() and torch.cuda.is_available()) else "cpu"
# =================================================

class BaseModelWrapper:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.ready = False

    def generate(self, pil_image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200):
        raise NotImplementedError


class LocalModelWrapper(BaseModelWrapper):
    def __init__(self, model_path: str, model_type: str, device_id: int = 0):
        super().__init__(model_type)
        self.model_path = model_path
        self.device_id = device_id
        self.model = None
        self.processor = None
        if model_path:
            self._load()

    def _load(self):
        if not HF_AVAILABLE:
            print("transformers/torch not available: cannot load HF model automatically.")
            return
        try:
            print(f"Loading processor/model from {self.model_path} onto GPU {self.device_id}...")
            if not os.path.exists(self.model_path):
                print(f"Model path does not exist: {self.model_path}")
                return
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                print("✓ Processor loaded.")
            except Exception as e:
                print(f"✗ Processor load failed: {e}")
                return

            try:
                if DEVICE == "cuda":
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map={"": self.device_id}
                    )
                else:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                print(f"✓ Model loaded on GPU {self.device_id}")
            except Exception as e:
                print(f"✗ Model load failed: {e}")
                return

            if self.model is not None:
                self.model.eval()
                self.ready = True
                print("✓ Model ready (eval mode).")
            else:
                print("✗ Model object is None.")
        except Exception as e:
            print(f"✗ Exception while loading model: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False

    def generate(self, pil_image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200):
        if not self.ready:
            status = "Image uploaded." if pil_image is not None else "No image uploaded."
            return (
                f"[{self.model_type}] Model not ready.\n"
                f"Prompt: {prompt}\n"
                f"{status}\n"
                "Please check model path / integrity."
            )
        try:
            # 构造多模态消息
            if pil_image is not None:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }]
            else:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if QWEN_VL_UTILS_AVAILABLE:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [pil_image] if pil_image is not None else None
                video_inputs = None

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # 统一设备：以模型首个参数为准，避免与分片/混合设备不一致
            model_device = next(self.model.parameters()).device
            moved_inputs = {}
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.device != model_device:
                    moved_inputs[k] = v.to(model_device)
                else:
                    moved_inputs[k] = v

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **moved_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=getattr(self.processor.tokenizer, "eos_token_id", None)
                )

            # 去掉前缀 token
            input_ids = moved_inputs["input_ids"]
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0] if output_text else ""
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Generation failed: {str(e)}"


class GoogleAPIModelWrapper(BaseModelWrapper):
    def __init__(self, api_key: str):
        super().__init__("Gemini 2.5 Pro API (French)")
        self.api_key = api_key
        self.ready = True

    def generate(self, pil_image: Optional[Image.Image], prompt: str, max_new_tokens: int = 200):
        if not self.api_key:
            return "Erreur: GOOGLE_API_KEY non défini dans l'environnement."
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            if not pil_image:
                return "Erreur: une image est requise pour générer un poème en français."
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = os.path.join(temp_dir, "image.jpg")
                pil_image.save(image_path, format="JPEG")
                image_file = client.files.upload(file=image_path)

                # Create dynamic prompt file content in-memory
                prompt_text_path = os.path.join(temp_dir, "prompt.txt")
                with open(prompt_text_path, "w", encoding="utf-8") as f:
                    f.write(prompt)
                prompt_file = client.files.upload(file=prompt_text_path)

                # Instructions file
                instructions_path = os.path.join(temp_dir, "instructions.txt")
                with open(instructions_path, "w", encoding="utf-8") as f:
                    f.write(
                        "You are to generate a French poem (e.g., Sonnet, Vers libre, Haïku) strictly in JSON.\n"
                        "Fields: create_condition (bool), poem_type, poem_title, poem_content, poem_explanation.\n"
                        "All textual content (title, content, explanation) must be in French.\n"
                        "Return ONLY a JSON object. Do not wrap in Markdown fences."
                    )
                instructions_file = client.files.upload(file=instructions_path)

                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[prompt_file, image_file, instructions_file]
                )

            raw_text = response.text or ""

            # Extract pure JSON
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            json_string = match.group(0) if match else ""
            if not json_string:
                return f"Erreur: réponse non JSON.\nTexte brut:\n{raw_text}"

            try:
                data = json.loads(json_string)
            except json.JSONDecodeError as e:
                return f"Erreur JSON: {e}\nContenu:\n{json_string}"

            # Normalize fields
            poem_data = data.get("generated_content", data)
            poem_type = poem_data.get("poem_type", "")
            poem_title = poem_data.get("poem_title", "")
            poem_content = poem_data.get("poem_content", "")
            poem_explanation = poem_data.get("poem_explanation") or poem_data.get("poem_analysis", "")

            # Repackage as JSON string (for formatter)
            normalized = {
                "poem_type": poem_type,
                "poem_title": poem_title,
                "poem_content": poem_content,
                "poem_explanation": poem_explanation
            }
            return json.dumps(normalized, ensure_ascii=False)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"API failure: {str(e)}"


# Initialize wrappers
api_wrapper = GoogleAPIModelWrapper(GOOGLE_API_KEY)
original_wrapper = LocalModelWrapper(ORIGINAL_MODEL_PATH, "Original Qwen2.5-VL-7B", device_id=0)
finetuned_wrapper = LocalModelWrapper(FINETUNED_MODEL_PATH, "LoRA Fine-tuned Qwen2.5-VL-7B", device_id=1)


def _plain_text_to_json(text: str) -> str:
    """
    将本地模型输出的普通法语诗歌文本包装成统一 JSON，方便统一渲染。
    规则：
      - 取首个非空行作为候选标题：若长度 <= 40 且不含超过2个标点且不是以小写字母开头，则作为标题。
      - 其余为正文。
      - 分析字段暂空。
    """
    raw = text.strip("\n")
    if not raw:
        return json.dumps({
            "poem_type": "Poème",
            "poem_title": "Poème sans titre",
            "poem_content": "",
            "poem_explanation": ""
        }, ensure_ascii=False)

    lines = [l.rstrip() for l in raw.splitlines()]
    # 去掉开头/结尾空行
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()

    if not lines:
        title = "Poème sans titre"
        body = ""
    else:
        candidate = lines[0].strip()
        # 标题启发式
        if (len(candidate) <= 40 and
            not candidate.endswith(("...", "：", ":", ";", ",")) and
            sum(1 for c in candidate if c in ".,;!?，。？！") <= 2 and
            (not candidate or candidate[0].isupper())):
            title = candidate
            body_lines = lines[1:]
        else:
            title = "Poème sans titre"
            body_lines = lines
        body = "\n".join(body_lines).strip("\n")

    data = {
        "poem_type": "Poème",
        "poem_title": title if title else "Poème sans titre",
        "poem_content": body,
        "poem_explanation": ""
    }
    return json.dumps(data, ensure_ascii=False)


def format_poem_output(result: str) -> str:
    """
    统一格式化：无论 API 还是本地模型，都尝试输出统一样式。
    """
    try:
        txt = result.strip()
        # 如果不是 JSON，尝试转换
        if not (txt.startswith("{") and txt.endswith("}")):
            txt = _plain_text_to_json(txt)

        poem_data = json.loads(txt)
        if "generated_content" in poem_data:
            poem_data = poem_data["generated_content"]

        poem_type = poem_data.get("poem_type", "Poème")
        poem_title = poem_data.get("poem_title", "Sans titre")
        poem_content = poem_data.get("poem_content", "")
        poem_explanation = poem_data.get("poem_explanation") or poem_data.get("poem_analysis", "")

        # 安全转义（保留换行后再转 <br>）
        poem_content_escaped = html.escape(poem_content).replace("\n", "<br>")
        poem_explanation_escaped = html.escape(poem_explanation)

        # 使用无缩进拼接，避免首行缩进导致排版异常
        parts = []
        parts.append("<div style=\"font-family:'Noto Sans','Noto Sans SC',sans-serif;"
                     "max-width:880px;margin:0 auto;padding:24px;"
                     "background:linear-gradient(135deg,#f9fafb 0%,#e3edf7 100%);"
                     "border-radius:16px;box-shadow:0 10px 28px rgba(0,0,0,0.08);\">")

        parts.append("<div style='text-align:center;margin-bottom:18px;'>"
                     f"<span style=\"display:inline-block;background:linear-gradient(135deg,#2563eb,#1e3a8a);"
                     "color:#fff;padding:6px 18px;border-radius:24px;font-weight:600;letter-spacing:.5px;"
                     f"font-size:0.95rem;\">{poem_type}</span></div>")

        parts.append(f"<h1 style=\"text-align:center;color:#1f2937;font-size:2.3rem;margin:0 0 28px;"
                     "font-weight:700;border-bottom:3px solid #fbbf24;display:inline-block;padding:0 0 10px;\">"
                     f"{html.escape(poem_title)}</h1>")

        parts.append("<div style=\"background:linear-gradient(to bottom,#fff,#f6f7f9);padding:38px 34px;"
                     "border-radius:12px;text-align:center;margin-bottom:30px;border:1px solid #e2e8f0;"
                     "position:relative;box-shadow:0 6px 18px rgba(0,0,0,0.05);\">"
                     "<div style=\"position:absolute;top:10px;left:14px;font-size:0.8rem;"
                     "color:#94a3b8;letter-spacing:1px;font-weight:600;\">VISION POÉTIQUE</div>"
                     "<div style=\"position:absolute;bottom:10px;right:14px;font-size:0.8rem;"
                     "color:#94a3b8;letter-spacing:1px;font-weight:600;\">IA CRÉATIVE</div>"
                     f"<div style=\"font-size:1.18rem;line-height:1.95;color:#374151;letter-spacing:0.3px;"
                     "font-family:'Georgia',serif;white-space:normal;text-align:left;\">"
                     f"{poem_content_escaped}</div></div>")

        parts.append("<div style=\"background:linear-gradient(90deg,#f1f5f9,#e0f2fe);padding:26px 28px;"
                     "border-radius:12px;border-left:6px solid #2563eb;box-shadow:0 4px 12px rgba(0,0,0,0.06);\">"
                     "<h2 style=\"margin:0 0 16px;color:#1f2937;font-size:1.35rem;display:flex;"
                     "align-items:center;gap:10px;\">"
                     "<span style=\"background:linear-gradient(135deg,#2563eb,#1e3a8a);color:#fff;"
                     "width:36px;height:36px;display:flex;align-items:center;justify-content:center;"
                     "border-radius:50%;font-weight:700;\">↯</span>"
                     "Analyse du poème (FR)</h2>"
                     f"<div style=\"font-size:1rem;line-height:1.7;color:#334155;background:rgba(255,255,255,0.7);"
                     "padding:16px 18px;border-radius:10px;font-family:'Noto Serif',serif;\">"
                     f"{poem_explanation_escaped if poem_explanation_escaped else '<i>(Aucune analyse fournie)</i>'}"
                     "</div></div>")

        parts.append("<div style=\"text-align:center;margin-top:32px;color:#64748b;font-size:0.9rem;"
                     "font-family:'Noto Sans',sans-serif;\">✦ Vision · AI French Poetry ✦</div>")
        parts.append("</div>")

        return "".join(parts)
    except Exception as e:
        return f"<pre style='white-space: pre-wrap; background:#f8fafc; padding:20px; border-radius:12px;'>Formatting error: {e}\n{result}</pre>"


def predict(image: Optional[Image.Image], prompt: str, max_new_tokens: int, mode: str):
    if image is not None and not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if mode == "Gemini 2.5 Pro API (French)":
        result = api_wrapper.generate(image, prompt, max_new_tokens)
    elif mode == "Original Qwen2.5-VL-7B":
        result = original_wrapper.generate(image, prompt, max_new_tokens)
    elif mode == "LoRA Fine-tuned Qwen2.5-VL-7B":
        result = finetuned_wrapper.generate(image, prompt, max_new_tokens)
    else:
        result = "Unknown mode."
    return format_poem_output(result)


def load_demo_example():
    return None, ("Please analyze the visual atmosphere of this image and compose a French poem (preferably a sonnet "
                  "or free verse) evoking mood, light, texture, and symbolic metaphors. Provide a concise French explanation.")


custom_css = """
body {
    background: linear-gradient(135deg,#eef2f7 0%,#d6e3f3 100%);
    font-family: 'Noto Sans', 'Noto Sans SC', sans-serif;
}
h1 {
    font-size: 2.6rem !important;
    color:#1f2937;
    text-shadow:2px 2px 4px rgba(0,0,0,0.1);
    margin-bottom:18px;
    border-bottom:3px solid #2563eb;
    padding-bottom:10px;
    font-weight:700;
}
button {
    font-size:1.05rem !important;
    border-radius:42px !important;
    padding:12px 26px !important;
    transition: all .25s ease !important;
    box-shadow:0 4px 10px rgba(0,0,0,0.08) !important;
    border:none !important;
    font-weight:600 !important;
}
button:hover {
    transform: translateY(-3px) !important;
    box-shadow:0 6px 16px rgba(0,0,0,0.15) !important;
}
.gr-box {
    border-radius:16px !important;
    box-shadow:0 4px 14px rgba(0,0,0,0.08) !important;
    border:1px solid #e2e8f0 !important;
    background:rgba(255,255,255,0.92) !important;
}
label {
    font-size:1rem !important;
    color:#334155 !important;
    font-weight:600 !important;
}
#generate_btn {
    background: linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    color:#fff !important;
}
#example_btn {
    background: linear-gradient(135deg,#38bdf8,#0ea5e9) !important;
    color:#fff !important;
}
#mode_btn {
    background: linear-gradient(135deg,#6366f1,#4338ca) !important;
    color:#fff !important;
    margin:8px;
}
#back_btn {
    background: linear-gradient(135deg,#fb7185,#f43f5e) !important;
    color:#fff !important;
    margin-bottom:12px;
}
#poem_output {
    font-size:1.1rem !important;
    line-height:1.7;
    color:#1e293b;
    background: linear-gradient(to bottom,#ffffff,#f1f5f9) !important;
    border-left:5px solid #2563eb !important;
    padding:18px !important;
    border-radius:14px !important;
}
footer {
    text-align:center;
    padding:18px;
    margin-top:26px;
    color:#64748b;
    font-size:0.9rem;
    border-top:1px solid #e2e8f0;
}
.output-title {
    font-size:1.4rem !important;
    color:#1e293b !important;
    margin-bottom:12px !important;
    text-align:center;
    border-bottom:2px solid #2563eb;
    padding-bottom:8px;
}
"""

with gr.Blocks(title="French Vision-to-Poetry Generator", css=custom_css) as demo:
    current_page = gr.State(value="main")

    # Main page
    with gr.Column(visible=True, elem_id="main") as main_page:
        gr.Markdown("""
        <div style="text-align:center; background: linear-gradient(135deg,#1e3a8a,#2563eb); 
                    padding:30px; border-radius:16px; margin-bottom:28px; color:white;">
            <h1 style="color:white; margin:0;">✨ Vision → French Poetry ✨</h1>
            <p style="font-size:1.05rem; max-width:800px; margin:14px auto 0;">
                Upload an image, craft a prompt, and let AI compose an original French poem with literary analysis.
            </p>
        </div>
        """)

        gr.Markdown("### Select Mode")

        with gr.Row():
            api_btn = gr.Button("🌐 Gemini 2.5 Pro API (French)", variant="primary", elem_id="mode_btn")
            original_btn = gr.Button("🧠 Original Qwen2.5-VL-7B", variant="primary", elem_id="mode_btn")
            finetuned_btn = gr.Button("🎨 LoRA Fine-tuned Qwen2.5-VL-7B", variant="primary", elem_id="mode_btn")

        gr.Markdown("### Mode Description")
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                <div style="background:#ffffffc9; padding:18px; border-radius:14px; box-shadow:0 4px 12px rgba(0,0,0,0.06);">
                    <h3 style="color:#2563eb;">🌐 Gemini 2.5 Pro API</h3>
                    <ul style="line-height:1.5;">
                        <li>Remote inference</li>
                        <li>Fast turnaround</li>
                        <li>Structured French JSON output</li>
                    </ul>
                </div>
                """)
            with gr.Column():
                gr.Markdown("""
                <div style="background:#ffffffc9; padding:18px; border-radius:14px; box-shadow:0 4px 12px rgba(0,0,0,0.06);">
                    <h3 style="color:#dc2626;">🧠 Original Model</h3>
                    <ul style="line-height:1.5;">
                        <li>General multimodal understanding</li>
                        <li>Generates French poetic text</li>
                        <li>Flexible style exploration</li>
                    </ul>
                </div>
                """)
            with gr.Column():
                gr.Markdown("""
                <div style="background:#ffffffc9; padding:18px; border-radius:14px; box-shadow:0 4px 12px rgba(0,0,0,0.06);">
                    <h3 style="color:#7e22ce;">🎨 Fine-tuned (LoRA)</h3>
                    <ul style="line-height:1.5;">
                        <li>Tuned for poetic elegance</li>
                        <li>Better metaphor density</li>
                        <li>Improved structural coherence</li>
                    </ul>
                </div>
                """)

        gr.Markdown("---")
        gr.Markdown("""
        <div style="text-align:center; padding:18px; background: linear-gradient(135deg,#f8fafc 0%, #eef2f7 100%);
                    border-radius:14px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
            Upload an evocative image → Provide a creative instruction → Receive a refined French poem + analysis.
        </div>
        """)

        gr.Markdown("""
        <footer>
            <p>French Vision-to-Poetry System © 2025</p>
            <p>Powered by Multimodal Large Language Models</p>
        </footer>
        """)

    # API Page
    with gr.Column(visible=False, elem_id="api_page") as api_page:
        gr.Markdown("# 🌐 Gemini 2.5 Pro API (French)")
        back_btn_api = gr.Button("← Back to Menu", elem_id="back_btn")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    input_image_api = gr.Image(type="pil", label="Upload Image")
                    prompt_box_api = gr.Textbox(
                        lines=4,
                        value=("Please compose a structured French poem (e.g., sonnet or free verse) based on the image. "
                               "Focus on mood, symbolism, chromatic atmosphere, and evoke sensory metaphors. "
                               "Return JSON with poem_type, poem_title, poem_content, poem_explanation all in French."),
                        label="Instruction"
                    )
                    max_tokens_api = gr.Slider(minimum=256, maximum=4096, step=8, value=2048,
                                               label="Max New Tokens (approx)")
                with gr.Row():
                    run_btn_api = gr.Button("✨ Generate", variant="primary", elem_id="generate_btn")
                    example_btn_api = gr.Button("📋 Load Example Prompt", elem_id="example_btn")
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_box_api = gr.HTML(elem_id="poem_output")

    # Original model page
    with gr.Column(visible=False, elem_id="original_page") as original_page:
        gr.Markdown("# 🧠 Original Qwen2.5-VL-7B")
        back_btn_original = gr.Button("← Back to Menu", elem_id="back_btn")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    input_image_original = gr.Image(type="pil", label="Upload Image")
                    prompt_box_original = gr.Textbox(
                        lines=4,
                        value=("Generate a refined French poem (prefer a classical form like 'sonnet' unless image "
                               "suggests otherwise). Use rich imagery, internal rhythm, and provide a French analysis."),
                        label="Instruction"
                    )
                    max_tokens_original = gr.Slider(minimum=256, maximum=4096, step=8, value=2048,
                                                    label="Max New Tokens (approx)")
                with gr.Row():
                    run_btn_original = gr.Button("✨ Generate", variant="primary", elem_id="generate_btn")
                    example_btn_original = gr.Button("📋 Load Example Prompt", elem_id="example_btn")
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_box_original = gr.HTML(elem_id="poem_output")

    # Fine-tuned model page
    with gr.Column(visible=False, elem_id="finetuned_page") as finetuned_page:
        gr.Markdown("# 🎨 LoRA Fine-tuned Qwen2.5-VL-7B")
        back_btn_finetuned = gr.Button("← Back to Menu", elem_id="back_btn")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    input_image_finetuned = gr.Image(type="pil", label="Upload Image")
                    prompt_box_finetuned = gr.Textbox(
                        lines=4,
                        value=("Compose an elegant French sonnet inspired by the image. Maintain structural integrity, "
                               "evocative metaphors, temporal atmosphere, and provide a French explanation."),
                        label="Instruction"
                    )
                    max_tokens_finetuned = gr.Slider(minimum=256, maximum=4096, step=8, value=2048,
                                                     label="Max New Tokens (approx)")
                with gr.Row():
                    run_btn_finetuned = gr.Button("✨ Generate", variant="primary", elem_id="generate_btn")
                    example_btn_finetuned = gr.Button("📋 Load Example Prompt", elem_id="example_btn")
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_box_finetuned = gr.HTML(elem_id="poem_output")

    def navigate_to_page(page):
        pages = {
            "main": [True, False, False, False],
            "api": [False, True, False, False],
            "original": [False, False, True, False],
            "finetuned": [False, False, False, True]
        }
        return [
            gr.update(visible=pages[page][0]),
            gr.update(visible=pages[page][1]),
            gr.update(visible=pages[page][2]),
            gr.update(visible=pages[page][3]),
            page
        ]

    api_btn.click(lambda: navigate_to_page("api"),
                  outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    original_btn.click(lambda: navigate_to_page("original"),
                       outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    finetuned_btn.click(lambda: navigate_to_page("finetuned"),
                        outputs=[main_page, api_page, original_page, finetuned_page, current_page])

    back_btn_api.click(lambda: navigate_to_page("main"),
                       outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    back_btn_original.click(lambda: navigate_to_page("main"),
                            outputs=[main_page, api_page, original_page, finetuned_page, current_page])
    back_btn_finetuned.click(lambda: navigate_to_page("main"),
                             outputs=[main_page, api_page, original_page, finetuned_page, current_page])

    example_btn_api.click(load_demo_example, outputs=[input_image_api, prompt_box_api])
    example_btn_original.click(load_demo_example, outputs=[input_image_original, prompt_box_original])
    example_btn_finetuned.click(load_demo_example, outputs=[input_image_finetuned, prompt_box_finetuned])

    run_btn_api.click(
        fn=predict,
        inputs=[input_image_api, prompt_box_api, max_tokens_api, gr.State("Gemini 2.5 Pro API (French)")],
        outputs=[output_box_api]
    )
    run_btn_original.click(
        fn=predict,
        inputs=[input_image_original, prompt_box_original, max_tokens_original, gr.State("Original Qwen2.5-VL-7B")],
        outputs=[output_box_original]
    )
    run_btn_finetuned.click(
        fn=predict,
        inputs=[input_image_finetuned, prompt_box_finetuned, max_tokens_finetuned,
                gr.State("LoRA Fine-tuned Qwen2.5-VL-7B")],
        outputs=[output_box_finetuned]   # ← 修正这里
    )

if __name__ == '__main__':
    demo.launch(share=True)