# GRPO - Group Relative Policy Optimization
* GRPO, or Group Relative Policy Optimization, is a reinforcement learning algorithm designed to enhance large language model (LLM) reasoning by evaluating groups of responses relative to each other, rather than relying on external critics.


# Why is GRPO important?
* In a nutshell, computational requirements drop significantly and it simplifies the RL process.
* GRPO basically cuts the compute requirements to do Reinforcement Learning from Human Feedback (RLHF) by about 50% compared to what was used for ChatGPT (PPO).
* When you take LoRA into consideration, this significantly opens the door to RL training for just about anyone that is GPU poor.

# Comparing PPO vs. GRPO
* [Resource](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)
* Proximal Policy Optimization (PPO) is the Reinforcement Learning (RL) technique predicted to be at the core of the ChatGPT model.
* OpenAI revealed in the InstructGPT paper that in order to create an LLM model that can follow human-like instructions and go beyond just predicting the next word, the **model training process requires you to collect a lot of labeled data.**
* The process looks like this:
  1. For a given user query, you have the LLM generate multiple candidate responses.
  2. Next, you have a human and/or LLM in the loop used to label and rank the outputs from "best to worst".
  3. The ranked outputs are then used as training data for a “rewards model”.
  4. The reward model's function is to calculate a “reward” for a new prompt that the model sees.
  5. The reward should represent how "good" this response is, given the user query.
 

![image](https://github.com/user-attachments/assets/74d958c4-1053-4088-9981-7c183139fd43)

Here is a breakdown of the chart above:

1. **Policy Model**
   * Fancy name for the current LLM you are training
2. **Reference Model**
   * A frozen version of the original LLM you are training
3. **Reward Model**
   * The model that was trained on human preferences (from the technique in InstructGPT above)
4. **Value Model**
   * A model that is trying to estimate the long term reward given certain actions
  
# Reducing Memory Usage with GRPO
* In PPO both the policy model and the value model have trainable parameters that need to be back-propagated in the neural network.
* Backprop requires a significant amount of memory.
* If you look at the diagram above, GRPO drops the value model.
* PPO has 4 LLMs in the mix, which all require substantial memory and compute.
* The value and reward models are typically of a comparable parameter count to the LLM you are training.
* The reference model is usually a frozen copy of the initial language model.

![image](https://github.com/user-attachments/assets/dac59f42-d25e-453e-978f-400b0a35461e)
