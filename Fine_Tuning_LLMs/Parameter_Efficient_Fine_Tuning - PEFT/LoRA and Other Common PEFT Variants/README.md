# LoRA and Other Common PEFT Variants


# LoRA: Low-Rank Adaptation
* Original arxiv paper by Hu et al in 2021: https://arxiv.org/abs/2106.09685


## LoRA Overview
* We know that for a 175B parameter model like (GPT-3), each parameter is in 32 or 16 bits or 2 bytes.
* Thus, it takes around 350 GB memory to run or fine tune which is almost impossible on most computers. 
* LoRA was proposed to solve this problem. 
   * The concept is to take a large matrices and project it into low rank matrices. 
   * So you take a 10x10 matrix —---into---> 10x1 and 1x10 matrices
      * Then M1 x M2 —> 10 x W matrix
      * 20 parameters result from decomposing large matrix (cross product is 20)

* LoRA paper showed that it performs “on par or better” fine tuning models such as these below and with no inference latency. 
   * RoBERTa
   * DeBERTa
   * GPT-2
   * GPT-3
 
## LoRA Mathematical Deep Dive
* I made this diagram with a deep dive into the low rank decomposition methods of LoRA and how it works.

![image](https://github.com/user-attachments/assets/cfc0684a-f2b4-49e1-a112-44d041bf347d)


## Why does LoRA work?
* Pre-trained LLMs exhibit intrinsic low dimensions. 
* Based on this, during fine tuning weight updates should be low rank matrices
* Low rank matrices A and B are learned 
* The model base layers are frozen. 

![image](https://github.com/user-attachments/assets/68a17e23-4710-4662-86d2-2a4ab9308cab)


## During Inference there is NO ADDITIONAL LATENCY!
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


# QLoRA
* QLoRA - Efficient Finetuning of Quantized LLMs
* Dettmers et al original paper in 2023: https://arxiv.org/abs/2305.14314
* Technique is the same as LoRA except adding Quantization.
   * Quantize the weights of a LoRA model. 
* As an example, GPT-3 is 175B parameters —> 2 bytes
   * LoRA
   * QLoRA 

## QLoRA with Phi-3 LLM as an example
* Phi-3 has 3.8B parameters or about 4B parameters. 
* Using LoRA —> 4B weights —> 16 bit format —>  8GB memory to load
* Less memory means you can fine-tune locally for less time and less cost. 
* QLoRA can represent this model in 2 different formats

1. 8 bit format
   * 4B x 1 byte —> 4 GB memory

2. 4 bit format
   * 4B x 0.5 byte —> 2 GB memory


* Remember, that quantization reduces the precision of the numbers (weights) in our model. 
  * Instead of using 32 bits to represent a number, we use 8 bits or even 4 bits. 
  * **This dramatically cuts down the memory usage.**
* Think of this as "Elephant sizes" as described by this [awesome blog post](https://medium.com/@shikharstruck/shrinking-elephants-a-funny-guide-to-4-bit-and-8-bit-quantization-for-llms-with-lora-ddf9f1a62070)


1. 32-bit Float (Full-Size Elephant)
  * High precision
  * large size

2. 8-bit Quantization (Smaller Elephant)
  * Reduced precision
  * smaller size

3. 4-bit Quantization (Tiny Elephant)
  * Even less precision
  * tiniest size!


## QLoRA Quantization techniques proposed
1. **4-bit NormalFloat (NF4)**
   * **optimized for normally distributed weights.**
   * *Existing post-training quantization (PTQ) solutions are primarily integer-based and struggle with bit widths below 8 bits. Compared to integer quantization, floating-point (FP) quantization is more flexible and can better handle long-tail or bell-shaped distributions, and it has emerged as a default choice in many hardware platforms.*
   * From the original paper: [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836)
   * Pre-trained neural network weights are usually normally distributed and centered around zero.
       * So, here is a very high probability of values occurring closer to zero rather than around -1 or plus 1.
       * However, standard quantization to int4 is not aware of this fact.
       * Thus it goes by the assumption that each of the 16 bins has an equal probability of getting the values.
       * NF4 considers the normal distribution of neural network weights.
         * This is what QLoRA does and it names it `k-bit NormalFloat`.
         * **In NormalFloat, the bins are weighted by the normal distribution and hence the spacing between two quantization values are far apart near the extremes of -1 or 1 but are closer together as we get closer to 0. Thus we need to account for the long tails with FP or floating point quantization.** [source](https://www.ai-bites.net/qlora-train-your-llms-on-a-single-gpu/)
  
![image](https://github.com/user-attachments/assets/aa46c8f8-753b-4d20-91f3-c0b57227fb73)


2. **Double Quantization**
   * reduces overall memory footprint

3. **Paged Optimizers**
   * **manages memory spikes in training process**
   * Gradient accumulation in forward and backward pass memory spikes occur.
   * These are distributed across CPU memory to handle this better.
   * As we can see from the original arxiv paper below:
  
 ![image](https://github.com/user-attachments/assets/f564b3cc-0d9c-4ab2-89c1-ceb931775b17)


