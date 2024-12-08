# Full Fine Tuning LLMs
* A repo devoted to full fine tuning of LLMs.


## `instruction fine tuning` Notebook Project Steps and Information
* Info
*  Project is full fine tuning of **TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T**
  * model card: https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T
* Dataset used: **HuggingFaceH4/no_robots**
  * dataset card: https://huggingface.co/datasets/HuggingFaceH4/no_robots

* Recommend running on GPU on runpod
1. Open Runpod instance
2. Choose RTX A6000 instance from “secure cloud tab”
3. Choose PyTorch 2.1.1 docker image 
4. Increase storage to 100 GB via “customize dep

* Next do this:
  * Open Terminal in Jupyter Lab
  * pip install -r requirements.txt file
  * log into weights and biases: `wandb login` 
    * This will prompt you for your API key via your profile. 
  * log into huggingface hub using the command `Huggingface-cli login`
  * This will prompt you to enter your token
    * Make sure it has “write access” to push models to hub
   


# Limitations of full fine tuning

1. Compute power
 * Large models need 10s to 100s of GPUs for memory
2. Storage
 * 70B llama model as example, full precision requires 280GB of storage.
 * Finetuned on 10 downstream tasks —> 2 TB storage needed!!!


## What makes a model large?
1. Number of parameters 
2. Precision of the data type
 * Example: Mistral-7B model with 7 billion parameters: 
    * FP32 —> 28 GB
    * FP16/BF16 —> ~15 GB
    * INT8 —> ~7 GB
    * INT4/NF4 —> ~3.5 GB


## Why is full finetuning expeinse?
Finetuning Mistral 7-B in mixed-precision using Adam Optimizer
1) Weights —> 2 bytes / parameter
2) Gradients —> 2 bytes / parameter
3) Optimizer state
4 byets / parameter (FP32 copy) + 8 bytes / parameter (momentum & variance estimates)
4) Total Training cost —> 16 bytes/parameter * 7 billion parameters = 112 GB
112GB doesnt even account for memory required for intermediate activations.
