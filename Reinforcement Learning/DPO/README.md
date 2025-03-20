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
