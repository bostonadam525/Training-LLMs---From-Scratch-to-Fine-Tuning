# Reinforcement Learning in LLM Fine-Tuning
* The ability to effectively fine-tune LLMs for specific tasks has become a crucial topic.
* Reinforcement Learning (RL) is an effective solution, with methods such as:
  1.  RLHF (Reinforcement Learning with Human Feedback)
  2.  PPO (Proximal Policy Optimization)
  3.  DPO (Distributed Proximal Policy Optimization)
  4.  KTO (Kahneman-Tversky Optimization)
  5.  ORPO (Odds-Ratio Preference Optimization)
 

# When do you use Supervised Fine Tuning (SFT) vs. Reinforcement Fine Tuning (RFT)?
* This chart and information below is from [Daily Dose of Data Science](https://www.dailydoseofds.com/)
* The process of Reinforcement Fine-tuning (RFT) allows you to transform any open-source LLM into a "reasoning powerhouse" without labeled data!
      * RFT utilizes an online “reward” approach—no static labels required.
      * The RFT model will explore various outputs, and use a Reward Function to score the correctness of those outputs.
      * Over time, the RFT model learns to generate higher-reward answers using GRPO.
* This is different from SFT uses static data and often memorizes answers.
* RFT which is an online method, will learn from rewards and explore new strategies that SFT can not. 


![image](https://github.com/user-attachments/assets/7989e2e3-062c-4c44-98f1-ad5187e9fd36)



# Overview of Post-Training LLM Reinforcement Learning and Optimization Techniques
* The chart below is from the arxiv paper published in February 20205 entitled "LLM Post-Training: A Deep Dive into Reasoning Large Language Models" by Kumar et al. [Link to paper](https://arxiv.org/html/2502.21321)
*  The chart gives us a great overview of specific models that were trained with some of these techniques. As the authors state, it is "a taxonomy of post-training approaches for LLMs (LLMs), categorized into Fine-tuning, Reinforcement Learning, and Test-time Scaling methods. We summarize the key techniques used in recent LLM models, such as GPT-4, LLaMA 3.3, and Deepseek R1."
*  Repo related to the chart and paper is [here](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training)

![image](https://github.com/user-attachments/assets/90eeaf8f-534d-4789-ba7f-17e14c1b9af3)


# Overview of LLM Fine-Tuning Methods

## 1. Supervised Fine Tuning (SFT)
* Supervision comes from instruction-response pairs.
  * Token level optimization of model compares predicted tokens to the actual tokens.
  * Uses cross-entropy or negative log likelihood loss.
* Why is this needed?
  * Ensurces model accurately understands and follows specific instructions.
 

```
Prompt or Instruction —> Gold-Standard Response —> Optimized LLM model
```

## 2. Reinforcement Learning from Human Feedback (RLHF)
* Supervision comes from a **reward model**. 
  * Algorithms such as PPO adjust policy (LLM weights) to **maximize the expected reward**. 
* Why is this needed?
  * Ensures that the model’s responses align with human values and preferences. 

* Steps to Build a RLHF model:

  * **Step 1:** Collect human feedback through response ranking and train reward model or quality model.
```
prompt or instruction —> ranked responses —> reward model trained
```

  * **Step 2:** Offline Reinforcement Learning with the reward model scoring LLM responses
  * This is the step where we actually train the model weights.
```
Prompt or Instruction —> LLM generates response —> RM scores response —> LLM optimizes model (e.g. PPO) —> iterate back to LLM generate response
```
  * The goal is to converge on the BEST model optimized weights. 

* **What is the Biggest problem with RLHF?**
  1. Expensive and Time consuming.
  2. You need to build a reward model which takes time and money and resources. 
  3. You need to continually update your RL model offline. 

## 3. Direct Preference Optimization (DPO)
* This approach came out of Stanford University.
* Supervision comes directly from instruction and preference pairs.
  * LLM has it’s own implicit reward model/signals —> this is related to generation likelihoods.
  * The goal is to increase the likelihood of “GOOD” responses and reduce likelihood of “BAD” responses.

* Why is this needed?
  1. More stable and simpler alternative to RLHF methods such as PPO. 
  2. Avoids need to train separate reward model.
    
* Positives of this method.
  1. More simple than RLHF.
  2. More stable than RLHF.
  3. Avoids need and time to train separate reward model(s).
 
```
Prompt or Instructions —> Good/Bad responses —> Optimized LLM model
```

# Important Notes

## **RLHF & DPO**
  * Dataset of instruction-response pairs
    * This Needs to be a diverse set of representative instructions.
    * Responses need to be variable —> can be human or LLM produced. 

  * Data production for RLHF and DPO
    * Annotator or model (RLAIF) ranks responses to instructions.
    * Rankings can be formed around specific dimensions (e.g. clarity)

  * Considerations
    * Preferences need to be CLEAR and Transitive. 
    * Subject preference dimensions lead to lower agreement. 
    * Need to be careful to avoid encoding human or annotator biases. 
```
Prompt or Instruction —> Annotatior-Ranked responses —> DPO | RLHF
```


# Other Methods

## 1. KTO (Kahneman-Tversky Optimization)
* This was inspired by "Prospect Theory" from the field of Economics. 
* KTO is a Modification of DPO which focuses on how humans perceive utility. 
* Dataset format is very simple --> Notes whether a response is GOOD or BAD. 
* **KTO is more robust to noisy data than DPO.** 

## 2. ORPO (Odds-Ratio Preference Optimization)
* Combines Supervised Fine Tuning (SFT) and preference optimization into a single step. 
* This is more computationally efficient than performing both steps separately.
* Outperforms DPO in some benchmarks. 
