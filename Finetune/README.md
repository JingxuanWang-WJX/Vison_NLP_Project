---
library_name: peft
license: other
base_model: /data/wtt/wjx_code/VLM/Qwen2.5-VL-7B-Instruct/
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: Qwen2.5-VL-7B-Instruct-Epoch=9
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Qwen2.5-VL-7B-Instruct-Epoch=9

This model is a fine-tuned version of [/data/wtt/wjx_code/VLM/Qwen2.5-VL-7B-Instruct/](https://huggingface.co//data/wtt/wjx_code/VLM/Qwen2.5-VL-7B-Instruct/) on the mllm_demo dataset.
It achieves the following results on the evaluation set:
- Loss: 1.4628
- Num Input Tokens Seen: 24407280

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 4
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- num_epochs: 9.0

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.55.0
- Pytorch 2.8.0+cu128
- Datasets 3.6.0
- Tokenizers 0.21.1