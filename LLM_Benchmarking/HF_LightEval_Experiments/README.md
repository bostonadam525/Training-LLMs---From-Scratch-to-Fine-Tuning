# Hugging Face Light Eval Experiments
* A repo for experiments with benchmarking LLMs on various tasks. 



# Overview
* Before using any LLM or let's say a multilingual LLM you may want to do an experiment to benchmark a few LLMs against each other using the Lighteval library from Hugging Face.



# Benchmark Process
* There are multiple frameworks to benchmark LLMs include `HELM` and `BIG-BENCH`.
* This better explains some of these frameworks: https://medium.com/@InsightfulEnginner/understanding-benchmarking-in-nlp-glue-superglue-helm-mmlu-and-big-bench-2e0a55b57d3b
* There is an entire repo of pre-defined benchmark tasks by the lighteval library that can be accessed here: https://github.com/huggingface/lighteval/blob/main/examples/tasks/all_tasks.txt
* Additional Evaluation resources
    * 1) LightEval getting started notebook: https://colab.research.google.com/drive/1iQnf_rTf2Bn9nC1gwHBzZ00NrTOMfBQq?usp=sharing#scrollTo=jWdS38syaipm
     * 2) MultiQ framework: [Evaluating the Elementary Multilingual Capabilities of Large Language Models with MultiQ](https://arxiv.org/html/2403.03814v1)
     * 3) [Quantifying Multilingual Performance of Large Language Models Across Languages](https://arxiv.org/html/2404.11553v2)
     * 4) SageMaker LiteEval: https://www.philschmid.de/sagemaker-evaluate-llm-lighteval



# Multilingual LLMs to consider benchmarking in the future
1. `lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual`
   * model card: https://huggingface.co/lightblue/DeepSeek-R1-Distill-Qwen-1.5B-Multilingual

2. `lightblue/suzume-llama-3-8B-multilingual`
    * model card: https://huggingface.co/lightblue/suzume-llama-3-8B-multilingual
  
3. `mistralai/Mixtral-8x7B-Instruct-v0.1`
    * model card: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
  
4. `Qwen-2.5-14B-Instruct`
    * model card: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
  
5. `Llama-3.1-8B-Instruct`
    * model card: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
