# Direct Preference Optimization - DPO

## Before DPO....
* Before DPO, we had to train a separate model to help us fine-tune called the reward model or RLHF model.
* We would sample outputs from the LLM and then prompt the reward model to give us a score for each output.
* The idea was simple:
    * Humans are expensive to have evaluate your LLMs outputs but the quality of your LLM will ultimately be determined by humans.
    * To keep costs down and maintain high quality, you would need to train the reward model to approximate human feedback.
    * This is why Proximal Policy Optimization (or PPO) came about and it 100% depends on the strength of your reward model.
 

![image](https://github.com/user-attachments/assets/2bc5d6c0-a3dd-4e37-bee0-b6992ec32c58)



## Enter DPO
* Direct Preference Optimization **completely eliminates the need for a rewards model!**
* This allows us to avoid the often high cost of training a separate rewards model and often DPO actually requires a lot less data than PPO [source](https://towardsdatascience.com/understanding-the-implications-of-direct-preference-optimization-a4bbd2d85841/)
* Original paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

![image](https://github.com/user-attachments/assets/214b5391-6cca-4071-9819-1b6dd1856d52)


# What about LoRA with DPO?

1. LoRA for Efficient Fine-Tuning:
  * LoRA (Low-Rank Adaptation) is a technique that allows for efficient fine-tuning of large language models (LLMs) by adding small, trainable LoRA modules to the existing model, rather than retraining the entire model. 

2. DPO and Model Alignment:
  * DPO is a method for aligning LLMs with human preferences by directly optimizing the model's rewards based on preference data. 

3. LoRA and DPO Combination:
  * Using LoRA with DPO allows you to fine-tune an LLM for a specific task or preference alignment without needing to store or train the entire model, which is especially beneficial for large models and limited computational resources. 

4. Parameter Efficiency:
  * LoRA focuses on updating a minimal number of parameters, while checkpoint methods often require full model retraining. 

5. Adaptability:
  * LoRA's low-rank updates allow for quick adaptations to new tasks, whereas checkpoint methods may involve more extensive modifications. 

## How LoRA with DPO works:
  * The actor is initialized by the reference model plus LoRA weights, where only the LoRA weights are trainable. 
  * This allows US to switch between the actor/reference models by simply enabling or disabling LoRA. 
  * There is no need to store two sets of LLM weights. 
