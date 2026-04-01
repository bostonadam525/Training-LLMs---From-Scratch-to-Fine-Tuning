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


---
# DPO - Constrastive Learning from Positive and Negative Samples

- Goal is to change the behavior and outputs of the model based on user preferences/feedback. 
- Use DPO loss function 
- Output is fine-tuned LLM with preference behaviors

## Modeling Preferences in LLM domain
- There are 2 very common statistical models used for this:
1. Bradley-Terry --> takes 2 items (chosen and rejected completions) -> and associated reward for each item as input. _Bradley-Terry model to express probabilities for pairwise comparisons between two completions_.
	- See recent paper: Fang et al, 2026. Recent advances in the Bradley–Terry Model: theory, algorithms, and applications. Link: https://arxiv.org/html/2601.14727v2
2. Plackett-Luce model --> extends the Bradley-Terry model to MULTIPLE comparisons. 


We can generally break this down into Pairwise vs. Listwise comparison:

1. Pairwise: Bradley-Terry (BT) Method: The Bradley-Terry model is the standard probabilistic approach for pairwise comparisons. - **Mechanism:** Assumes each item has a latent worth (score) and models the probability that item i beats item j as a logistic function of their scores.
- **Best for:** When you have many comparisons of only two items (e.g., sports, A/B testing).
- **Limitations:** Assumes independence of irrelevant alternatives (IIA)—meaning the preference between A and B is independent of the presence of C. It also requires transitivity (if A>B and B>C, then A>C), which may not hold in real-world scenarios.
- **Modern Use:** Extensively used in training Large Language Models (LLMs) via Direct Preference Optimization (DPO) and Elo ratings for competitive gaming.


1. Listwise: Plackett-Luce (PL) Model: The Plackett-Luce model is a generalization of Bradley-Terry for ranking more than two items.
- **Mechanism:** Models the probability of a ranked list as a sequence of choices, where the top-ranked item is chosen from the pool, then the second from the remaining, and so on.
- **Best for:** When data consists of ordered lists (e.g., "Rank these 5 items").
- **Advantage:** More efficient than breaking lists into pairs, as it uses the full ordering information.
- **Reduction:** When applied to pairs, the Plackett-Luce model reduces exactly to the Bradley-Terry model.

3. Pairwise vs. Listwise vs. Rank-based Approaches

- **Pairwise:** Breaks all preferences into individual comparisons (A vs B). Highly robust, handles sparse data well, but can be computationally expensive if every combination is needed.
- **Listwise:** Treats the entire ranking (A>B>C) as one observation. It directly optimizes ranking-specific metrics like NDCG (Normalized Discounted Cumulative Gain). Often performs better than pairwise when high-quality ranking data is available.
- **Rank-based (Mallows Model):** Instead of assigning worth (scores) like BT/PL, these models identify a "modal" (center) ranking and determine how quickly the probability of a ranking decreases as it moves away from the center.

4. Alternatives and Advancements

- **Thurstonian Models:** Similar to BT, but assume latent utilities follow normal distributions rather than Gumbel distributions.
- **Mixture Models:** To handle population heterogeneity (different users having different preferences), mixture models can combine multiple BT or PL models.
- **SpringRank:** Ranks nodes by treating the comparison network as a physical system of springs, useful for noisy data.
- **BT-SBM (Stochastic Block Model):** Embeds the BT model within a block model to cluster items and rank them, useful for identifying groups of similar quality.

Sources:
- Hermes et al, 2024. Joint Learning from Heterogeneous Rank Data. Link: https://arxiv.org/html/2407.10846v1#:~:text=Aside%20from%20the%20Thurstonian%20model,data%20that%20we%20focus%20on.
- Liu et al, 2024. LiPO: Listwise Optimization through Learning-to-Rank. Link: https://arxiv.org/html/2402.01878v1
- Santilabel et al, 2025. The Bradley-Terry Stochastic Block Model. Link: https://arxiv.org/html/2511.03467v1#:~:text=Abstract,-Report%20issue%20for&text=The%20Bradley%E2%80%93Terry%20model%20is,more%20competitive%20in%20recent%20years.
- Sun et al, 2024. Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives. Link: https://arxiv.org/html/2411.04991v1#:~:text=While%20the%20latter%20corresponsds%20to%20the%20BT,be%20the%20Thurstonian%20model%20(Thurstone%2C%201927)%20.
- Truong et al, 2025. Machine Learning from Human Preferences. Link: https://mlhp.stanford.edu/


## DPO - Loss Function: What is it doing?

- DPO minimizes the contrastive loss which penalizes negative responses and encourages positive responses. 
- DPO loss is a cross entropy loss on the reward difference of a "re-parameterized" reward model
	- Cross entropy loss is a mechanism to quantify how well a model’s predictions match the actual outcomes, rewarding the model for assigning higher probabilities to correct answers. Because it uses the logarithm function, cross entropy loss is more sensitive to changes for low-confidence predictions on the correct class. 
	- This encourages the model to quickly resolve uncertainty by increasing confidence in correct predictions, while heavily penalizing confident mistakes.

The key intuition is:

- **The more confident the model is in predicting the correct outcome, the lower the loss.**
- The more confident the model is in predicting the wrong outcome, the higher the loss.

- Source_1: https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5
- Source_2: https://cameronrwolfe.substack.com/p/direct-preference-optimization
- Original DPO paper: Rafailov et al, 2024. Direct Preference Optimization: Your Language Model is Secretly a Reward model. Link: https://arxiv.org/html/2305.18290v3

https://substackcdn.com/image/fetch/$s_!yQz2!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7107abbb-358e-48d4-a200-64ca6b5d1d72_2050x1092.png<img width="1456" height="776" alt="image" src="https://github.com/user-attachments/assets/cc63ca86-d69e-4848-a5ac-d0b46f50798e" />



- Beta is very important hyperparameter tuned during training. HIGHER beta --> more important log diff is. 
- Chosen Reward: Reference model (copy of original model)
- Rejected Reward: Reparameterization of reward model


## Best Use Cases for DPO?
1. Changing Model Behavior
- Making small modifications of model responses
	- Identity
	- Multilingual
	- Instruction following
	- Safety
	- ...etc.


1. Improving model capabilities 
- Better than SFT in improving model capabilities due to contrastive nature. 
- Online DPO is better for improving capabilities than offline DPO. 
	- Google DeepMind paper: "Understanding the performance gap between online and offline alignment algorithms" (2024). Link: https://arxiv.org/html/2405.08448v1



## Principles of DPO Data Curation
- Common methods for high-quality DPO data curation:

1. Correction
- Generate responses from original model as negative, make enhancements as positive response. 
	- Example: I'm Llama (Negative) --> I'm Claude (Positive)

1. Online/On-Policy
- Your positive & negative example can both come from your model's distribution. One may generate multiple responses from the current model for the same prompt, and collect the best response as positive sample and the worst response as negative. 
	- One can choose best/worst response based on reward functions/human judgement




- Avoid Overfitting! 
	- DPO is doing reward learning which can easily overfit to some shortcut when the preferred answers have shortcuts to learn compared with the non-preferred answers. 
	- Example: when positive sample always contains a few special words while negative samples DO NOT. 


