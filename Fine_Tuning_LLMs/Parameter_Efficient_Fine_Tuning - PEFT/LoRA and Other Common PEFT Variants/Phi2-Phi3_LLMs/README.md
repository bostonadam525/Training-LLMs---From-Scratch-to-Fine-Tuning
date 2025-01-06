# Phi2 and Phi3 LLM Models Fine Tuning

## Phi-3
* Original technical report: https://arxiv.org/abs/2404.14219
* 3.8 billion param LLM
* Performance rivals Mistral 8x7B and GPT-3.5
* 69% on MMLU benchmark
* Main innovation is the dataset they used for training --> was a “scaled up” version of the data used for phi2 which contained filtered public web and synthetic data.
* Phi3 vision also introduced with 4.2 billion parameters for image and text prompts. 


## Tokens to Parameters
* If an LLM model has X parameters. You usually need `20 * X` tokens to train a model. 
* As an example, lets say you have 4 billion parameter model. 
      * This means that you have: `4B * 20 = 80 Billion` tokens at minimum to train this model. 
* The 20:1 token to parameter ratio for optimal training is known as the **“Chinchilla paradigm shift”** which was established in 2022. 
* Early scaling laws (Kaplan et al., 2020) established power-law relationships between model size, data, and performance.

      * The Chinchilla paradigm shift (2022) introduced the 20:1 token-to-parameter ratio for optimal training.
          * [Chinchilla paper](https://arxiv.org/abs/2203.15556)
      * Post-Chinchilla developments saw “overtraining” beyond the 20:1 ratio, yielding performance gains.
      * Recent models like Llama-3 pushed token-to-parameter ratios to 200:1, challenging previous assumptions.
      * Inference scaling (OpenAI’s o1 model, 2024) emerged as a new direction, focusing on optimising inference-time compute for improved reasoning.
