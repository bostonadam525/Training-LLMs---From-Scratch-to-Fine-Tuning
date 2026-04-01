# Post-Training of LLMs


---
## What is Post-Training? 

1. Take a pre-trained Base Model which predicts next word/token and teach it to learn responses from curated data. 
2. This usually includes Instruct/Chat Model such as:
   - Respond to instructions: Q-> what is the capital of Connecticut? A -> Hartford
3. Then there is Continual Post-Training -- the goal here is for "changing behaviors or enhancing model capabilities"
	 - This is a further customization step --> specialized domains often take this step.
   - This might mean training on specific coding languages, science, math, history, etc..
  

---
## Pre-Training Methods
- LLMs are typically **pretrained on massive unlabeled text corpora**—often hundreds of billions to trillions of tokens—from sources such as **Common Crawl, The Pile, GitHub, books, and Wikipedia**. 
- For **autoregressive pretraining** (used in GPT-style models), the model learns to predict each next token from the previous tokens in a sequence. 
- Mathematically, for tokens \(x_1, x_2, \dots, x_T\), the sequence probability is factorized as \(p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})\), and training minimizes the **negative log-likelihood / cross-entropy loss**: \(\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})\). 
	- For example, for “I like pizza,” the loss is \(\mathcal{L} = -\log p(\text{I}) - \log p(\text{like} \mid \text{I}) - \log p(\text{pizza} \mid \text{I like})\). 
	- While next-token prediction is the standard objective for autoregressive LLMs, related pretraining variants include **prefix language modeling**, **multi-token prediction**, and auxiliary objectives, while non-autoregressive approaches include **masked language modeling** (BERT) and **denoising/span corruption** (T5/BART).

---
# Post-Training Methods

## Loss Functions for LLMs

### Recent Research
1. This is a recent paper on the REINFORCE method. The authors propose "REINFORCE with Group Relative Advantage (RGRA), a simplified variant that retains group-relative advantage estimation but removes PPO-style clipping and policy ratio terms...Their "results suggest that simpler REINFORCE-based approaches can effectively enhance reasoning in LLMs, offering a more transparent and efficient alternative
to GRPO."
  * Paper: Carrino et al, 2026. ARE COMPLICATED LOSS FUNCTIONS NECESSARY FOR TEACHING LLMS TO REASON? Link: https://arxiv.org/pdf/2603.18756

2. [Everything You Need to Know About LLMs — Part 3: Loss Functions, Tokenization, and Evaluation.](https://osintteam.blog/everything-you-need-to-know-about-llms-part-3-062cc8e7de8f)

3. [Emergent Mind - Contrastive Triplet Loss Overview](https://www.emergentmind.com/topics/contrastive-triplet-loss)

4. Loss functions review: https://vinija.ai/concepts/loss/

5. [Comparing Contrastive and Triplet Loss: Variance Analysis and Optimization Behavior](https://arxiv.org/html/2510.02161v2)

---
## Post-Training Method 1: Supervised Fine-Tuning (SFT)
- Supervised / Imitation Learning 
- in this common method we take labeled-prompt-response pairs and fine-tune such as:
	- Prompt: Explain what machine learning is
	- Response: ML is the....
	- Could be 1K to 1B tokens or more 
- Mathematical formula for training loss: min pie - log pie (Response | Prompt)
	- ONLY train on tokens for responses NOT tokens for prompt. 

---
## Post-Training Method 2: Direct Preference Optimization (DPO)
- Prompt + Good and Bad Responses
- Example:
	- Prompt: Explain ML to me...
	- Good Response: ML is the...
	- Bad Response: Sorry I didnt get that
- Train on again less tokens than pre-training: ~1K-1B tokens 
- Loss function:
	- min pie - log sigma (Beta (log(pie(Good R | Prompt)/pie ref(Good R | Prompt) - log (pie(Bad R | Prompt) / pie ref(Bad R | Prompt)))))

---
## Post-Training Method 3: Online Reinforcement Learning
- Prompt + Reward Function
	- Example:
		- Prompt: Explain ML to me...
		- Response: ML is the....
		- Reward: 2.5
	- Train again less tokens than pre-train: ~1K to 10M prompts 
- Loss function: max pie Reward(Prompt, Response(pie))

---
# Post-Training Requires Getting 3 Elements Correct

## 1. Data & Algorithm co-design:
- SFT (non-instruction vs. instruction?)
- DPO
- Reinforce/RLOO/RLAIF --> NOTE: Reinforce is a specific algorithm!
- GRPO
- PPO
- .....


## 2. Reliable and Efficient Library
- Huggingface TRL
- OpenRLHF
- VeRL
- Nemo RL


## 3. Appropriate Evaluation Suite
- incomplete list of LLM evals:
	- 1) human preferences for chat
		- Chatbot Arena -- vote for which model is better
	- 2) LLM as a judge for chat
		- Alpaca Eval
		- MT Bench
		- Arena Hard V1/V2
	- 3) Static Benchmarks for Instruct LLM
		- LivecodeBench
		- AIME 2024/2025
		- GPQA
		- MMLU Pro
		- IFEval
	- 4) Function Calling & Agent
		- BFCL V2/V3
		- NexusBench V1/V2
		- TauBench -- multi toos 
		- ToolSandbox -- multi tools

NOTE: Its easy to improve any one of the benchmarks. It is MUCH HARDER to improve WITHOUT DEGRADING OTHER DOMAINS. 


---
# Do you really need post-training?

| Use Cases                                                                                                                     | Methods                                                                                                         | Characteristics                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Folllow a few instructions (e.g. do not mention xyz)                                                                          | Prompting                                                                                                       | Simple but brittle: models may not always follow all instructions they are trained on!                            |
| Query real-time databases or knowledge base                                                                                   | RAG or Hybrid Search                                                                                            | Adapts to rapidly-changing knowledgebases                                                                         |
| Create a Medical LLM                                                                                                          | Continual Pre-training (learn domain knowledge) + Post-training (learn specific instructions, behaviors, etc..) | Inject large-scale domain knowledge (>1B tokens) NOT seen during pre-training                                     |
| Follow 20+ instructions closely; Improve targeted capabilities ("Create a strong Python / function calling / reasoning model) | Post-training                                                                                                   | Reliably changes model behavior & improve targeted capabilities; May degrade other capabilities if not done right |
|                                                                                                                               |                                                                                                                 |                                                                                                                   |





