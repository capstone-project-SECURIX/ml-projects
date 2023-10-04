# Ass1-Classification Algorithm

Topic : EmotionandsentimentanalysisoftweetsusingBERT

Research Paper Link : chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ceur-ws.org/Vol-2841/DARLI-AP_17.pdf

Code-Colab-link : https://colab.research.google.com/drive/14aLspmK2MbTUmECW-GvT5M4kQBaAvbKn?usp=sharing


# Model 1 : Llama 2 7B chat finetune App
   - demo link : https://huggingface.co/atharvapawar/Securix_GPT_Neo
```

```




# Model 1 : Securix Llama 2 Fine Tuning using QLora 
   - demo link : https://huggingface.co/atharvapawar/Securix_GPT_Neo
```
```



# Model 1 : Securix Fine Tune GPT Neo 
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
      Model Type: The model you've described is based on GPT-2 (Generative Pre-trained Transformer 2). It's a variant of the GPT-2 model architecture.

      Model Size: The model you've mentioned has 12 layers, 12 attention heads, and an embedding dimension of 768. These are common specifications for GPT-2 models.

      Dropout: Dropout is used in various parts of the model, including the attention dropout (attn_pdrop), embedding dropout (embd_pdrop), and residual dropout (resid_pdrop). These are techniques to prevent overfitting during training.

      Vocabulary Size: The model's vocabulary size is 50,257 tokens. This defines the number of unique tokens the model can generate text for.

      Prompt Generation: You've mentioned "GPT-Neo for Stable Diffusion Prompt Gen," but it's not clear how this model configuration relates to that specific task. GPT-2 can be used for generating text based on prompts, but the provided configuration doesn't include details specific to the "Stable Diffusion Prompt Gen" task.