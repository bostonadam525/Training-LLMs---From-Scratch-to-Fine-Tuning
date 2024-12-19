# LoRA and Other Common PEFT Variants


## LoRA: Low-Rank Adaptation
* Original arxiv paper by Hu et al: https://arxiv.org/abs/2106.09685

### Why does LoRA work?
* Pre-trained LLMs exhibit intrinsic low dimensions. 
* Based on this, during fine tuning weight updates should be low rank matrices
* Low rank matrices A and B are learned 
* The model base layers are frozen. 

![image](https://github.com/user-attachments/assets/68a17e23-4710-4662-86d2-2a4ab9308cab)


### During Inference there is NO ADDITIONAL LATENCY!
* During training:
  * Step 1: Train adapters adapted on your task. 
  * Step 2: Merge adapter weights inside the base model and use it as a standalone model. 

![image](https://github.com/user-attachments/assets/33873af8-a8a1-40ee-9940-907adb394577)



## LoRA Variants
1. LoHA
  * LoRA with Hadamard Product
2. LoKr
  * LoRA with Kronecker Product
3. AdaLoRA
  * Adaptive budget allocation such that important layers having HIGHER RANK (more parameters) while pruning less important layers OUT. 

![image](https://github.com/user-attachments/assets/ad8b3f26-cda2-4a81-8a78-87e20e0f5ac6)


## LoRA Finetuning Costs
* If we were finetuning Mistral-7B in mixed-precision using Adam Optimizer. 
  * Trainable parameters: 21,549,136
  * All parameters: 7,263,322,192
  * Trainable %: 0.296

### Breakdown of finetuning costs using LoRA
* Weights: 2 bytes/parameter
* Gradients: 2 bytes/parameter
* Optimizer state: 4 bytes/paramaeter (FP32 copy) + 8 bytes/parameter (momentum & variance estimates)
* Total training cost: 16 bytes/parameter * 7 billion parameters * 0.0029 + 14 = 112 * 0.00296 + 14 GB ~14.4 GB

