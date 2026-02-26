# Knowledge Editing for LLMs
* This repo is devoted to code, information, and techniques related to "knowledge editing" for LLMs.


---
# What is Knowledge Editing for LLMs?
* Knowledge editing for Large Language Models (LLMs) is the process of precisely updating, correcting, or removing specific factual information within a pre-trained model without requiring computationally expensive full retraining.
* It aims to fix outdated or inaccurate information while ensuring the model maintains its overall performance and capability.
---
# TL;DR
* This is a very interesting topic!
* While most "knowledge editing" of LLMs consists of "textual gradients" (e.g. prompt tuning, TextGrad, DSPy, etc.) vs. RAG vs. PEFT fine tuning, there are a plethora of other techniques I honestly didn't know about until I stumbled upon them in my own research.
* Although the biggest problem with knowledge editing is "Catastrophic Forgetting", it does have some use cases and continues to evolve. 
* This is one of the original papers [published in 2022 out of Stanford NLP](https://arxiv.org/abs/2110.11309)
* Some additional interesting starter resources:
  * [KnowledgeEditingPapers GitHub](https://github.com/zjunlp/KnowledgeEditingPapers?tab=readme-ov-file#-why-knowledge-editing)
  * [KnowEdit Framework](https://zjunlp.github.io/project/KnowEdit/)


---
## Main Types of Knowledge Editing Techniques
* Knowledge editing methods are generally categorized by how they interact with the model's parameters or structure: 

1. **Locate-and-Edit (Parameter-Modifying)**
   * Identifies specific neurons or weights associated with a particular piece of knowledge and directly modifies them to reflect new information, such as Rank-One Model Editing (ROME) or FT-L.

2. **Meta-Learning Methods**
   * Trains a separate "editor" model (a hyper-network) to learn how to update the weights of an LLM efficiently, as used in Model Editor Networks with Gradient Decomposition (MEND).

3. **External Memory/Adapter-Based Methods**
   * Stores new knowledge in auxiliary, external data structures (like a cache or small adapter layers) that are queried alongside the original, unchanged model parameters, such as SERAC.

4. **Representation Editing**
   * Dynamically alters the internal hidden states or activations during the inference process to correct knowledge, rather than changing the model's weights permanently.

5. **In-Context Editing**
   * Utilizes prompting techniques to provide the correct knowledge within the input prompt, guiding the model to the accurate answer without changing internal parameters
---
* [Source: KnowledgeEditingPapers GitHub](https://github.com/zjunlp/KnowledgeEditingPapers?tab=readme-ov-file#-why-knowledge-editing)

<img width="819" height="414" alt="image" src="https://github.com/user-attachments/assets/f38c079c-5e83-4d6e-b735-f53355628e7b" />




---
## Key Aspects of Knowledge Editing

1. **Efficiency**
   * Designed to be fast and cost-effective compared to retraining.

2. **Precision**
   * Targeted updates that change only the intended fact without affecting unrelated knowledge (avoiding "ripple effects").

3. **Generalization**
   * Ensuring the edited fact is applied to variations of the same query.

4. **Types of Knowledge Edited**
   * Includes updating factual, conceptual, or behavioral information (e.g., correcting misinformation or removing bias).
  


---
# Resources
* [An Initiative to Explore, Understand, and Advance Editing Techniques for LLMs and Agents](https://model-editing.github.io/)
* [Bodapati, 2023. Teaching speech recognizers new words — without retraining](https://www.amazon.science/blog/teaching-speech-recognizers-new-words-without-retraining)
* [Cohen et al, 2023. Evaluating the Ripple Effects of Knowledge Editing in Language Models](https://arxiv.org/abs/2307.12976)
* [Datta et al, 2026. Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis](https://arxiv.org/html/2602.20207v1)
* [Ge et al, 2024. Time Sensitive Knowledge Editing through Efficient Finetuning](https://arxiv.org/abs/2406.04496)
* [Guo et al, 2024. Two Optimizers Are Better Than One: LLM Catalyst for Enhancing Gradient-Based Optimization](https://arxiv.org/html/2405.19732v2)
* [Gupta et al, 2024. Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://arxiv.org/abs/2401.07453)
* [He et al, 2025. Knowledge Updating? No More Model Editing! Just Selective Contextual Reasoning](https://arxiv.org/abs/2503.05212)
* [Hossain et al, 2025. Investigating Model Editing for Unlearning in Large Language Models](https://arxiv.org/abs/2512.20794)
* [Hsueh et al, 2024. Editing the Mind of Giants: An In-Depth Exploration of Pitfalls of Knowledge Editing in Large Language Models](https://arxiv.org/abs/2406.01436)
* [Jiang et al, 2024. Learning to Edit: Aligning LLMs with Knowledge Editing](https://arxiv.org/abs/2402.11905)
* [KnowledgeEditingPapers GitHub](https://github.com/zjunlp/KnowledgeEditingPapers?tab=readme-ov-file#-why-knowledge-editing)
* [KnowEdit Framework](https://zjunlp.github.io/project/KnowEdit/)
* [Kumar, 2024. Exploring Pitfalls of Knowledge Editing in Large Language Models](https://medium.com/@techsachin/exploring-pitfalls-of-knowledge-editing-in-large-language-models-a0ab043909d0)
* [Li et al, 2024. Model Editing for LLMs4Code: How Far are We?](https://arxiv.org/abs/2411.06638)
* [Lucky, 2025. Editing Neural Knowledge: How to Rewrite What Language Models ‘Know’](https://medium.com/@luckyikenkwocha/editing-neural-knowledge-how-to-rewrite-what-language-models-know-8c104c27687b)
* [Memory Augmentation and Editing Techniques in LLMs](https://nace.ai/research/memory-augmentation-and-editing-techniques-in-large-language-models)
* [Mitchell et al, 2022. Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309)
* [Park et al, 2025. MAKE: Memory-Associated Knowledge Editing](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.26/132652/MAKE-Memory-Associated-Knowledge-Editing)
* [Soliman, 2024. Updating large language models by directly editing network layers](https://www.amazon.science/blog/updating-large-language-models-by-directly-editing-network-layers)
* [Wang et al, 2024. WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/abs/2405.14768)
* [Wei et al, 2024. Stable Knowledge Editing in Large Language Models](https://arxiv.org/abs/2402.13048)
* [Yao et al, 2023. Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172)



