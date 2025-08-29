# 华中科技大学人工智能与自动化学院视觉与自然语言处理（2025春）课程作业



#### 针对课程任务要求，本项目提出一种基于大模型协作的诗歌标注微调框架：

> [!IMPORTANT]
>
> 1. 搭建调用 Gemini-2.5-Pro API，以Prompt Engineering为核心构建自动化诗歌数据标注pipeline。总共对2369张图片进行标注，成功标注2345张，分离不适合作诗图片24张。
>
> 2. 利用成功标注2345张的2345张图片，使用监督微调（Supervised Fine-Tuning）中的LoRA微调方法，依托开源项目LLama_Factory（[hiyouga/LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024)](https://github.com/hiyouga/LLaMA-Factory)）
>
>    对Qwen2.5-VL-7B进行LoRA微调，得到效果极佳的微调模型。
>
> 3. 我们搭建了一个专注于根据图片生成诗歌的Web UI界面，包含调用Gemini-2.5-Pro API、加载原始Qwen2.5-VL-7B模型、加载微调Qwen2.5-VL-7B模型三种方式根据用户输入图片以及用户输入prompt生成所需要的诗歌，同时支持用户根据自己的需要进行个性化诗歌定制。



#### 环境配置与安装

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
pip install -q -U google-genai
pip install qwen_vl_utils
```



#### 调用 Gemini-2.5-Pro API 标注数据

##### 第一步：启动 generate.py 标注数据。

```
git clone https://github.com/lawrencewjx58/Vison_NLP_Project.git
cd Vison_NLP_Project
python generate.py
```

##### 第二步：若 generate.py 因网络连接等原因中断，请运行 delete.py 删除已标注的图片以避免重复标注。

```
python delete.py
```

##### 第三步：当标注完所有数据后，运行 merge.py，将多次标注得到的输出文件融合为一个输出文件，方便后续处理。

```
python merge.py
```

##### 第四步：运行 count.py，对全部标注数据进行统计，查看重复标注的数据以及标注失败的数据。

```
python count.py
```



#### 使用 Gemini-2.5-Pro API 标注数据对Qwen2.5-VL-7B进行LoRA微调

##### 第一步：运行 transform.py，将全部标注数据转化为微调所需的数据格式，本项目选用的模板为 mllm_demo。

```
python transform.py
```

##### 第二步：安装 Qwen2.5-VL-7B 原始模型。

```
huggingface-cli download --resume-download Qwen/Qwen2.5-VL-7B-Instruct --local-dir {your_path} --local-dir-use-symlinks False --token {your_token}
```

##### 第三步：启动 LLama_Factory 进行 LoRA 微调，微调参数设置为 learning_rate=5e-5, epoch=9.0, LoRA Rank r=64, LoRA Alpha=128, Drop_out=0.05。

```
cd LLaMA-Factory
llamafactory-cli webui
```

##### 我们的 LoRA 微调模型参数已开源至 Hugging Face ([lawrencewjx58/Qwen2.5-VL-7B_LoRA_image2poem · Hugging Face](https://huggingface.co/lawrencewjx58/Qwen2.5-VL-7B_LoRA_image2poem))



#### 运行Web UI界面可视化展示成果

```
cd Vison_NLP_Project
python UI_Web.py
```

