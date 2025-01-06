# Parameter Efficient Fine Tuning - PEFT
* This is a great depiction of the various PEFT methods from a recent paper by Xu et al. entitled "Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment".
  * arxiv paper link: https://arxiv.org/abs/2312.12148

![image](https://github.com/user-attachments/assets/e9b7cc3d-b076-4b66-af15-c92a2b5b62b6)


## Original PEFT paper
* Original paper published in 2019 by Houlsby et al: https://arxiv.org/abs/1902.00751
* At this time there was no GPT-3, only T5, BERT, GPT-1, no LLMs. 
   * This technique was originally proposed and utilized for fine tuning small language models like those I just mentioned. 
* Only a small sample/subset of parameters are fine tuned.
* Original PEFT paper results:
   * PEFT method was originally developed for BERT transformers on 26 text classification benchmarks. 
   * On GLUE benchmark they attained 0.4% of performance of full-fine tuning adding only 3.6% of params per task as compared to full fine tuning 100% parameters. 



## Why do we need PEFT?
1) As models become larger and larger, full fine-tuning will become infeasible to train on consumer hardware. 
2) In addition, storing and deploying fine-tuned models independently for each downstream task becomes very expensive
  * This is because fine-tuned models are the same size as the original pretrained model! 


## So what exactly is PEFT?
* These approaches allow us to fine-tune large pretrained LLMs on specific downstream tasks while requiring **significantly fewer parameters** than full fine-tuning. 

## What is the goal of PEFT?
* Achieve comparable or even better performance than full fine-tuning while requiring less computation and memory resources. 

## How does PEFT actually work?
* PEFT approaches will **ONLY fine-tune a small number of (extra) parameters** while **freezing MOST of the parameters of the pretrained foundation/base LLMs.**
* This will GREATLY DECREASE the computational and storage costs. 
* PEFT adapts to downstream tasks while **reducing the risk of "catastrophic forgetting”**.
  * As a reminder, “catastrophic forgetting” is when a model can forget what it was trained on during fine tuning. PEFT can prevent this phenomenon.
* Prevents Overfitting
  * PEFT is able to do this by **ONLY fine tuning a fraction of model parameters.** 


# Types of PEFT Techniques
1. **Soft Prompting Approaches**
  * Prompt Tuning
  * Prefix Tuning
  * Adaptation Prompt (Additive fine-tuning or Adapters)

2. **LoRA (and its variants) — most popular**
  * The overall concept is to reaparameterize the model into low rank matrices (also known as Reparameterization). 
  * This will reduce the number of overall trainable parameters while at the same time allowing the model to work with high dimensional parameters of the pretrained LLMs.
  * The other variants of LoRA include but are not limited to: 
      * QLoRA (quantized LoRA)
      * AdaLoRA
      * LoftQ
      * LoHA
      * LoKR

3. **Adapters**
   * Houlsby et al. introduce the concept of Adapters in their 2019 paper entitled: "Parameter Efficient Transfer Learning for NLP". Arxiv link: https://arxiv.org/abs/1902.00751.
     * In Adapter based learning **only the new parameters** are trained while the original LLM is frozen, thus we learn only a very small proportion of parameters of the original LLM.
     * Some of the main takeaways from this paper:
       1. Adapters attain high performance
       2. Permits training on tasks sequentially, that is, it does not require simultaneous access to all datasets
       3. Adds only a small number of additional parameters per task.
       4. Model retains memory of previous tasks (learned during pre-training).
   * Some of these methods include but are not limited to: 
      * 1) Bottleneck Adapters
          * Based on concept of adding fully connected neural networks between model layers. 
      * 2) IA3 - "Infused Adapter by Inhibiting and Amplifying Inner Activations"

# How would you use PEFT Methods in practice?
* The most common way to do this is via Hugging Face and their open source ecosystem. 
* Hugging Face has a specific library called “PEFT” 
    * The main focus is on Usability and Simplicity
    * Interoperability and Integration with Hugging Face ecosystem. 

