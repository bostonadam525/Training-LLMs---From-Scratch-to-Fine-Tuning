# Prompt Learning PEFT Techniques

# Soft Prompt Based Methods for PEFT
* This technique maps the problem of finding discrete hard prompt to a continuous soft prompt.

1. Prompting in Zero and Few shots have shown remarkable performance.
  * However, when there are A LOT OF training examples available, optimizing discrete natural language prompts is VERY CHALLENGING.

2. **Soft or Continuous Prompt methods overcome the limitation mentioned above**
  * Instead of optimizing the discrete natural language prompt...
      * learning them in a latent space by introducing trainable prompt tokens or prefix tokens and prepending them to input embeddings or intermediate hidden states during fine-tuning improves upon this limitation.
   

## Prompt Tuning
* This was first introduced by the 2021 paper by Lester et al. entitled: **"The Power of Scale for Parameter-Efficient Prompt Tuning”**
  * arxiv link: https://arxiv.org/abs/2104.08691
* Important figure 2 from this paper:
![image](https://github.com/user-attachments/assets/4e609617-a29e-4c5a-90b2-65b7ad8b3976)


* **Main takeaways from paper**:
1) Full fine tuning model in the paper had 11B parameters with 3 downstream tasks (A, B, C) all which also have 11B parameters. 
  * Thus the full fine tuning of this model will take 44GB of storage (11 * 4 = 44) for the pre-trained model plus the 3 individual tasks.
  * Every model would require tens to hundreds of DPOs
2) **The opposite approach is “Prompt Tuning”**
  * Here we can see that only a small fraction of the model parameters are fine tuned in batches of 20K parameters each.
  * The process is as follows:
      * a) For each batch you have input embeddings.
      * b) We prepare the soft prompt from the token embeddings, then pass it through pre-trained model to get predictions based on predictions & labels
      * c) We then perform back propagation and update the weights and gradients
          * This is done for each of the soft prompt input embeddings to optimize the fine tuning process. 
      * d) The most important thing to note here is there is **MINIMAL STORAGE —> total is 11GB as opposed to 44GB for the full fine tuning.**


## Prefix Tuning - Another Soft Prompt Fine Tuning Method
* The task here is taking the input which is a tuple of tokens from a knowledge base which **includes the relationship between entities.** 
* Original arxiv paper: (Prefix-Tuning: Optimizing Continuous Prompts for Generation)[https://arxiv.org/abs/2101.00190]
* Paper results
   * Applying prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization the authors found that by the LLM learning only 0.1% of parameters, prefix-tuning is able to obtain comparable performance in the full data setting. 
   * This method also OUT PERFORMS fine-tuning in low-data or data sparsity settings.
   * This method also extends well to unseend data during training or fine-tuning. 
* **Example: Harry Potter + Hogwarts —> Education is the relationship between them.**
   * You want to fine tune the model to generate the proper relationship between these tokens or entities.
   * Example from original paper:
 ![image](https://github.com/user-attachments/assets/f8ec1740-0ea0-40db-9e62-a32e46dae2f2)

* **The difference between this and Prompt Tuning:**
   * Here we are prepending soft prompt embeddings. 
   * While this is similar to Prompt Tuning, however, here the prompt embeddings are in each of the ATTENTION layers of the model. 
   * As an example, if GPT has 16 layers, then EACH LAYER has tunable layers to insert prompt embeddings.
   * Figure 1 from the original paper:

![image](https://github.com/user-attachments/assets/cf74d094-240e-4a54-a147-8ff4555a11bb)
