# Image-to-Poetry Framework Based on Large-Model Collaboration and Fine-Tuning

## Supported Languages: Chinese, French

#### This project proposes a collaborative large-model framework for poem annotation and fine-tuning:

> [!IMPORTANT]
>
> 1. We build an automated poetry data annotation pipeline driven by Prompt Engineering and powered by the Gemini-2.5-Pro API. A total of 2,369 images were processed: 2,345 successfully annotated, and 24 filtered out as unsuitable for poem generation.
>
> 2. Using the 2,345 successfully annotated image–poem pairs, we perform Supervised Fine-Tuning (SFT) with the LoRA method via the open-source project LLaMA-Factory ([hiyouga/LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024)](https://github.com/hiyouga/LLaMA-Factory)), applying it to Qwen2.5-VL-7B and obtaining a high-quality fine-tuned model.
>
> 3. We provide a Web UI dedicated to generating poetry from images. It supports three generation modes: invoking the Gemini-2.5-Pro API, using the original Qwen2.5-VL-7B model, and loading the LoRA fine-tuned Qwen2.5-VL-7B model. Users can upload an image and supply a custom prompt to steer style, theme, tone, or form, enabling personalized poetic output.

#### Environment Setup & Installation

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
pip install -q -U google-genai
pip install qwen_vl_utils
```

#### Annotating Data via Gemini-2.5-Pro API

##### Step 1: Run generate.py to start batch annotation.

```bash
git clone https://github.com/JingxuanWang-WJX/Vison_NLP_Project.git
cd Vison_NLP_Project
python generate.py
```

##### Step 2: If generate.py stops (e.g., network failure), run delete.py to remove already annotated images and avoid duplicate processing.

```bash
python delete.py
```

##### Step 3: After all annotation batches complete, run merge.py to consolidate multiple output files into a single unified file.

```bash
python merge.py
```

##### Step 4: Run count.py to produce statistics: detect duplicated annotations and failed cases.

```bash
python count.py
```

#### LoRA Fine-Tuning Qwen2.5-VL-7B with Gemini-Annotated Data

##### Step 1: Run transform.py to convert all annotated data into the required fine-tuning format. Template used: mllm_demo.

```bash
python transform.py
```

##### Step 2: Download the original Qwen2.5-VL-7B model.

```bash
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir {your_path} --local-dir-use-symlinks False --token {your_token}
```

##### Step 3: Launch LLaMA-Factory and configure LoRA fine-tuning. Recommended hyperparameters:
- learning_rate = 5e-5
- num_train_epochs = 9.0
- LoRA rank (r) = 64
- LoRA alpha = 128
- LoRA dropout = 0.05

```bash
cd LLaMA-Factory
llamafactory-cli webui
```

##### Released LoRA Weights (Chinese Poetry)
[lawrencewjx58/Qwen2.5-VL-7B_LoRA_image2poem_Chinese](https://huggingface.co/lawrencewjx58/Qwen2.5-VL-7B_LoRA_image2poem_Chinese)

##### Released LoRA Weights (French Poetry)
[lawrencewjx58/Qwen2.5-VL-7B_LoRA_image2poem_French](https://huggingface.co/lawrencewjx58/Qwen2.5-VL-7B_LoRA_image2poem_French)

#### Running the Web UI for Visualization

```bash
cd Vison_NLP_Project
python UI_Web.py
```


