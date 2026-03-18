# Fine Tuning LLMs - Full Pipeline
* Repo by Adam Lang
* This repo walks through various fine tuning approaches for different use cases.

---
# LLM Training Pipeline
* This is a [great review of fine tuning techniques](https://www.emergentmind.com/topics/fine-tuning-approaches)
* Excellent review paper by CeADAR Connect Group
* [The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and Opportunities (Version 1.1)](https://arxiv.org/html/2408.13296v3)


—————————
# Overview
---
## 1. **Unsupervised Pre-training or Self Supervised Learning**
    1. Objective: next-token prediction (language modeling)
    2. Process is unsupervised or self-supervised —> self-supervised because label is provided while training by the model. Train on LARGE CORPUS of data — usually internet data. 
    3. Result: foundation base model —> e.g. Llama, Mistral, GPT, DeepSeek, Gemini, etc.
    4. After this stage —> model developed knowledge and grammar understanding. However, it still lacks ability to follow instructions, may be biased in tone or outputs (not domain specific), not able to produce structured results.

---
## 2. **SFT — Supervised Fine Tuning**

### 1. Parameter Level — 2 parts
        1. Full fine-tuning —> Train ALL PARAMETERS (weights + biases)
            1. Requires HUGE GPU MEMORY!!!
            2. Requires MULTI-GPU Setup (parallel)
        2. Partial fine-tuning — 2 methods. Tuning subset of all parameters.
            1. “Old school method” —> 2 approaches (usually CNNs or early state transformers such as BERT, T5, BART)
                1. freeze all layers and train last output layer 
                2. Freeze a starting layer + re-train some last layer of the model
            2. PEFT - Parameter Efficient Fine-Tuning — (might work on single-GPU and smaller VRAM)
                1. LoRA - low rank adaptation
                2. QLoRA — quantized LoRA
                    1. Lower precision
                    2. memory efficient loading
                3. DoRA — [see paper](https://arxiv.org/abs/2402.09353)
                4. Adapter Layers
                5. BitFit — [see paper](https://arxiv.org/abs/2106.10199)
                    1. BitFit is best used when you want to maximize efficiency or have limited GPU memory while still obtaining high-quality task adaptation.
                    2. “Bias-term fine tuning”. This is a parameter-efficient fine-tuning (PEFT) method that adapts pre-trained transformer-based models to new tasks by updating only a tiny fraction of the total parameters. 
                    3. Freezes Most Parameters: BitFit freezes all pre-trained weights in the transformer-encoder and updates only the bias vectors and the final task-specific linear classifier layer.
                    4. Extremely Sparse Training: Typically, only 0.08% to 0.1% of the original model's parameters are tuned, making it highly memory-efficient.
                    5. Comparable Performance: Despite the sparse updates, BitFit achieves performance on par with (or sometimes better than) full-parameter fine-tuning on small-to-medium-sized datasets.
                    6. Ideal for Deployment: Because most parameters are frozen, it enables easy multi-task deployment, where a single base model can be used with different, tiny sets of bias terms for different tasks.
                    7. Faster Training & Lower Memory: It reduces the storage footprint and computation costs, making it suitable for resource-constrained scenarios. 
                6. IA3 
                        1. IA3 — [see paper](https://huggingface.co/papers/2205.05638)
                        2. Concept is to improve LoRA. 
                        3. Ultra-Efficient Scaling: Instead of training new layers or updating all weights, IA3 injects learned vectors to rescale inner activations (specifically in attention and feedforward modules), making it incredibly parameter-efficient.
                        4. Frozen Base Model: The original pre-trained model weights remain frozen, allowing for multiple, small, task-specific IA3 adapters (often <0.01% of parameters) to be used with one base model.
                        5. Zero Inference Latency: Because the learned scales are vectors rather than added layers, they can be "baked" or merged into the base model's weights after training, resulting in no extra computation during inference.
                        6. Strong Performance: Despite having fewer trainable parameters than LoRA, IA3 provides comparable or better results to full fine-tuning for many downstream tasks.
                        7. High Learning Rates: IA3 supports significantly higher learning rates than LoRA, allowing for faster convergence in fewer training steps. 
                7. Prefix tuning
                8. Prompt tuning (not really PEFT, but related)



——-

### 2. Data (level) Driven fine tuning approaches
        * Based on data preparation and features NOT on parameters. 
        * SFT Methods
            1. Non-instructional fine-tuning
            2. Instruction fine-tuning 



——
### 3. Alignment with human feedback (Preference Based Learning)
* Alignment of responses from LLM to human preferences
* 2 main methods
    * 1) RLHF 
        * Based mainly on PPO (proximal preference optimization)
        * Reinforcement learning (popularized by OpenAI)
    * 2) DPO
        * Direct preference optimization 
        * This is “Supervised Learning”
        * Dataset structured:
            * Question | Response | Feedback (pos, neutral, negative)

—--
# Supervised Fine-Tuning (Normal Fine-Tuning)

    1. This is often called “Normal fine-tuning” or “Domain adaptation”.
        1. Dataset: plain text or company documents
        2. Use Case: fine-tune LLM to be domain specific such as: medicine, legal, finance, etc.
        3. Goal: improve model’s understanding, knowledge, grammar, and ability to work in a specific domain’s natural language. 
        4. Output style: text continuation —> not necessarily  capable of following  instructions though. 
        5. Meaning: model is learning statistical distribution of text — not question-answering behaviors.
        6. Example: 

—--
## Instruction Fine-Tuning
* This is a sub-class of SFT (supervised fine tuning)
* Dataset: follows instructions + response format. 
* Use case: Chatbots, Q&A systems, tutors, coding assistants, customer support, etc.
