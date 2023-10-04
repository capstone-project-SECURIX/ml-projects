# Ass1-Classification Algorithm

# Comparative Study of Language Models

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
  - [Model 1: Llama 2 7B chat finetune App](#model-1-llama-2-7b-chat-finetune-app)
  - [Model 2: Securix Llama 2 Fine Tuning using QLora](#model-2-securix-llama-2-fine-tuning-using-qlora)
  - [Model 3: Securix Fine Tune GPT Neo](#model-3-securix-fine-tune-gpt-neo)
- [Comparative Analysis](#comparative-analysis)
- [Conclusion](#conclusion)

## Introduction
This README provides a comparative study of three different language models and their specifications. Each model serves specific purposes and has its unique characteristics.

## Models

### Model 1: Llama 2 7B chat finetune App
- **Description**: This model is fine-tuned for chat-based applications.
- **Application**: Chatbot development, conversational AI.
- **Use**: Generates human-like responses in chat scenarios.
- **Limitation**: Limited to chat-based use cases.

#### Specifications
- **Architecture**: LlamaForCausalLM
- **Hidden Size**: 4096
- **Intermediate Size**: 11008
- **Attention Heads**: 32
- **Hidden Layers**: 32
- **Maximum Position Embeddings**: 4096
- **Vocabulary Size**: 32000
- **Torch Data Type**: float16
- **Transformers Version**: 4.31.0

### Model 2: Securix Llama 2 Fine Tuning using QLora
- **Description**: This model is fine-tuned for specific tasks using QLora.
- **Application**: Customized language generation tasks.
- **Use**: Generates text with specialized fine-tuning.
- **Limitation**: Requires expertise in QLora fine-tuning.

#### Specifications
- **Architecture**: LlamaForCausalLM
- **Hidden Size**: 4096
- **Intermediate Size**: 11008
- **Attention Heads**: 32
- **Hidden Layers**: 32
- **Maximum Position Embeddings**: 2048
- **Vocabulary Size**: 32000
- **Torch Data Type**: bfloat16
- **Transformers Version**: 4.31.0
- **LORA Alpha**: 16
- **LORA Dropout**: 0.1

### Model 3: Securix Fine Tune GPT Neo
- **Description**: Fine-tuned GPT-2 model for various text generation tasks.
- **Application**: Text generation, content creation.
- **Use**: Generates text with diverse applications.
- **Limitation**: Model size is 124M, which might limit complexity.

#### Specifications
- **Architecture**: GPT2LMHeadModel
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Hidden Layers**: 12
- **Maximum Context Length**: 1024
- **Vocabulary Size**: 50257
- **Torch Data Type**: float32
- **Transformers Version**: 4.20.1

## Comparative Analysis
- Model 1 is specialized for chat-based applications, while Model 2 is customizable with QLora fine-tuning.
- Model 3 offers versatility in text generation tasks.
- Model 1 and 2 have a lower maximum position embedding compared to Model 3.
- Model 3 has a smaller vocabulary size but offers good text diversity.
- Model 2 incorporates QLora fine-tuning for specific tasks.

## Conclusion
The choice of language model depends on the specific application and requirements. Model 1 and 2 cater to specialized needs, while Model 3 offers versatility in text generation tasks. Consider the specifications and fine-tuning options when selecting a model.




------------------------------------------------------------------------------------------
# Deep Details
------------------------------------------------------------------------------------------

# Model 1 : Llama 2 7B chat finetune App

## Model Overview
```
      Library: PEFT
      Language: English (en)
      Pipeline Tag: Text2Text Generation
      Tags: Code Generation (cod)
```
   - Fine Tuning Code Link (Google Colab) : https://colab.research.google.com/drive/1GdzZ-Ush-nUFB8HmM46qPrv4Ng3U499U?usp=sharing
   - demo link : https://huggingface.co/atharvapawar/Llama-2-7b-chat-finetune-app
   - Model Configration:
      ```
      {
      "_name_or_path": "NousResearch/Llama-2-7b-chat-hf",
      "architectures": [
         "LlamaForCausalLM"
      ],
      "bos_token_id": 1,
      "eos_token_id": 2,
      "hidden_act": "silu",
      "hidden_size": 4096,
      "initializer_range": 0.02,
      "intermediate_size": 11008,
      "max_position_embeddings": 4096,
      "model_type": "llama",
      "num_attention_heads": 32,
      "num_hidden_layers": 32,
      "num_key_value_heads": 32,
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.31.0",
      "use_cache": true,
      "vocab_size": 32000
      }
      ```
   - Points:
      -- Model Name: NousResearch/Llama-2-7b-chat-hf
      -- Architecture: LlamaForCausalLM
      -- Hidden Size: 4096
      -- Intermediate Size: 11008
      -- Number of Attention Heads: 32
      -- Number of Hidden Layers: 32
      -- Maximum Position Embeddings: 4096
      -- Model Type: Llama
      -- Vocabulary Size: 32000
      -- Torch Data Type: float16
      -- Transformers Version: 4.31.0
      -- Pretraining TP: 1

   - Api Interface:

      ```
      #Lang : Python
      import requests

      API_URL = "https://api-inference.huggingface.co/models/atharvapawar/Llama-2-7b-chat-finetune-app"
      headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

      def query(payload):
         response = requests.post(API_URL, headers=headers, json=payload)
         return response.json()
         
      output = query({
         "inputs": "Can you please let us know more details about your ",
      })
      ```



# Model 2 : Securix Llama 2 Fine Tuning using QLora 

## Model Overview
```
      Library: PEFT
      Language: English (en)
      Pipeline Tag: Text2Text Generation
      Tags: Code Generation (cod)
```

   - Fine Tuning Code Link (Google Colab) : https://colab.research.google.com/drive/1wAQ8Kg_0NXH1sChfq_ZF6nrZJAYRYisp?usp=sharing 
   - demo link : https://huggingface.co/atharvapawar/securix_Llama-2-7B-Chat-GGML
   - Model Configration:
```
{
  "_name_or_path": "TheBloke/Llama-2-7B-fp16",
  "architectures": ["LlamaForCausalLM"],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.31.0",
  "use_cache": true,
  "vocab_size": 32000,
  "auto_mapping": null,
  "base_model_name_or_path": "TinyPixel/Llama-2-7B-bf16-sharded",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 64,
  "revision": null,
  "target_modules": [
    "q_proj",
    "v_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```
   - Training Procedure
      The model was trained with a quantization configuration using the bitsandbytes quantization method. Some key configurations include:

      Quantization Method: bitsandbytes
      Load in 8-bit: False
      Load in 4-bit: True
      LLM Int8 Threshold: 6.0
      LLM Int8 Skip Modules: None
      LLM Int8 Enable FP32 CPU Offload: False
      LLM Int8 Has FP16 Weight: False
      BNB 4-bit Quant Type: nf4
      BNB 4-bit Use Double Quant: False
      BNB 4-bit Compute Dtype: float16

   - Points:
      -- Qlora: Qlora is not a widely recognized term or acronym in the context of natural language processing or machine learning. Without additional context, it's challenging to provide a specific explanation for Qlora. It's possible that Qlora could be a custom module or component specific to the "Llama-2" model or the project it is associated with. If you have more context or information about Qlora, please provide it, and I'll do my best to assist you further.

      -- GGML: "GGML" is not a standard acronym or term in the field of natural language processing or machine learning. It's possible that GGML could be an abbreviation or a specific component or technique used in the same project as the "Llama-2" model. If you have more information about GGML or its context, please share it, and I'll try to provide a more accurate explanation.

   - Api Interface:

      ```
      #Lang : Python
      import requests

      API_URL = "https://api-inference.huggingface.co/models/atharvapawar/securix_Llama-2-7B-Chat-GGML"
      headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

      def query(payload):
         response = requests.post(API_URL, headers=headers, json=payload)
         return response.json()
         
      output = query({
         "inputs": "Can you please let us know more details about your ",
      })
      ```



# Model 3 : Securix Fine Tune GPT Neo 

## Model Overview
```
      Library: PEFT
      Language: English (en)
      Pipeline Tag: Text2Text Generation
      Tags: Code Generation (cod)
```

   - Fine Tuning Code Link(Kaggle) : https://www.kaggle.com/code/mrappplg/fine-tune-gpt-neo-for-stable-diffusion-prompt-gen 
   - demo link : https://huggingface.co/atharvapawar/Securix_GPT_Neo
   - Model Configration:
```
{
  "_name_or_path": "aitextgen/pytorch_model_124M.bin",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "line_by_line": true,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": null,
  "n_layer": 12,
  "n_positions": 1024,
  "n_vocab": 50257,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torch_dtype": "float32",
  "transformers_version": "4.20.1",
  "use_cache": true,
  "vocab_size": 50257
}
```
   - Points:
      1. Model Type: The model you've described is based on GPT-2 (Generative Pre-trained Transformer 2). It's a variant of the GPT-2 model architecture.

      2. Model Size: The model you've mentioned has 12 layers, 12 attention heads, and an embedding dimension of 768. These are common specifications for GPT-2 models.

      3. Dropout: Dropout is used in various parts of the model, including the attention dropout (attn_pdrop), embedding dropout (embd_pdrop), and residual dropout (resid_pdrop). These are techniques to prevent overfitting during training.

      4. Vocabulary Size: The model's vocabulary size is 50,257 tokens. This defines the number of unique tokens the model can generate text for.

      5. Prompt Generation: You've mentioned "GPT-Neo for Stable Diffusion Prompt Gen," but it's not clear how this model configuration relates to that specific task. GPT-2 can be used for generating text based on prompts, but the provided configuration doesn't include details specific to the "Stable Diffusion Prompt Gen" task.
   - Api Interface:

      ```
      #Lang : Python
      import requests

      API_URL = "https://api-inference.huggingface.co/models/atharvapawar/Securix_GPT_Neo"
      headers = {"Authorization": "Bearer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

      #enter your Read API Key

      def query(payload):
         response = requests.post(API_URL, headers=headers, json=payload)
         return response.json()
         
      output = query({
         "inputs": "Can you please let us know more details about your ",
      })
      ```