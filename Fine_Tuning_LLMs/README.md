# Fine Tuning LLMs
* A repository by Adam Lang

## What is finetuning? 
* At a high-level this is the process of adapting pre-trained general-purpose model by training it further to:
1. Specialize in given task(s) of choice by gaining domain knowledge.
2. Change model behavior to provide more consistent outputs and better control the tone and style of the generative outputs.
3. Align the LLM model to be more “helpful”, “harmless”, “honest” and hopefully prevent hallucinations. 


## Why is Finetuning even necessary?
* The most common “Recipe” for getting state of the art results (SOTA) in NLP:
1. Pretraining on web-scale datasets
2. Fiinetuning on downstream task of interest

* As an example, if we were to ask a foundation LLM this question: **“What are the top 5 places to visit in Burlington, VT”**
  * The LLM may respond with: “What are the top 5 places to visit in the USA?”
      * This may be because the LLM does not have enough domain specific knowledge to answer the question with context specific information.

* A fine-tuned model should and would provide more specific results to the domain of the question.


## Prompt Engineering vs. Fine Tuning
* This is one of the "age-old" questions, which is better and why?

* **Prompt Engineering**
  * Low cost up-front, no additional training data or compute power.
  * Overall less customizable because the prompts can only do so much to “steer” the Foundation LLM model in the “right direction” without updating the training data. 


* **Finetuning**
  * High cost up front!!
  * More data and more compute power!!!
  * High level of customization for obvious reasons.
 
* Comparison of the two (source: Analytics Vidyha)
![image](https://github.com/user-attachments/assets/d6136343-9d75-48f6-998e-cf3e859394a4)


## Advantages of Finetuning
1. Performance 
  * Accuracy and conistency much higher than prompting alone.
2. Data Privacy and Security (ownership of your data)
3. Lower operational costs
4. Increased Scalability and Reliability

## What is Instruction Finetuning?
* A subset of  ﻿Finetuning
* Training pre-trained LLMs to follow instructions with human feedback.
* This is from the original Open AI paper:
  * 1) Dataset of tuples consisting of: 
    * Instruction
    * Inputs
    * Response
  * 2) Answer/mimic like a human in order to behave like a chatbot
  * 3) Examples of models Instruction Finetuned:
    * ChatGPT
    * SharChat
    * Zephyr


# Breakdown of the Finetuning Process
* Data quality is the KEY during finetuning.
* Here are the primary steps:
 * 1) Data gathering and Preprocessing
 * 2) Training
 * 3) Evaluation
* This process is iterative and repeated multiple times with finetuning the parameters and new data.


## Pre-Training vs. Finetuning
* This is again a very common debate and topic of discussion.
* A major difference to remember:
   * Pre-Training uses self-supervised learning — there is NO annotated or labeled data. 
   * Finetuning uses Supervised learning WITH annotated and labeled data which is highly recommended. 
Self-supervised learning is also supported.
* Here is an excellent comparison of the 2 methods (source: Analytics Vidhya):
![image](https://github.com/user-attachments/assets/7279f696-7013-4f1a-a54e-8be0d434b648)

### 1. Data gathering and preprocessing 
* Dataset collection is paramount to a good result.

1. **ANNOTATED DATA BY SKILLED HUMANS not by an LLM or other source** 
 * High quality and diverse datasets
 * Examples of such datasets used for building conversational agents:
   * LIMA (“less is more for alignment” — high quality data)
   * OpenAssistant Conversations Dataset
   * No Robots

2. **LLM assisted Data Generation**
 * "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
   * link to paper: https://arxiv.org/abs/2212.10560
 * Best practices:
   * a) Start with “seed” instructions
   * b) Use LLM to generate more instructions
   * c) Filter them out for quality and diversity 
   * d) Rinse and repeat (iterate! Iterate! iterate!)

 * Use prompt template or LLM to generate instructions and input/output pairs for things such as: 
   * Corporate documents
   * Meeting Notes
   * Blogposts
   * Wikipedia Articles
   * …etc…
 * Example models:
   * Alpaca
   * Ultrachat
   * CodeAlpaca
 * External Datasets to consider:
   * FAQs
   * Transcribed customer support conversations, meetings, podcasts
   * Conversations from social media, discord, slack, etc..
   * Textbook questions and answers
   * Github issues/PR conversations 
   * ….anything that is relevant to the domain of your LLM fine-tuning

#### Best Practices to follow during Data Gathering
1. Quality
  * High-quality datasaets is the key!!
  * Remember: Avoid “Garbage in —> Garbage out”
2. Quantity
  * For instruction finetuning, data in order of thousands of samples is a good starting point.
  * However, the more data the better!
3. Diversity
  * Diverse tasks enable generalization and serendipity 
  * This can be diversity in language, length of input and output responses, the list goes on. 
4. Source of Data
  * Human annotated data is the GOLD STANDARD.
  * LLM generated data often has subtle patterns that can often hinder optimal fine tuning of an LLM.

#### Data Collection Pipeline
* Collect instruction dataset in Tuples of:
  * `(Instruction, Input, Output)`
* Format and Concatenate Samples
* Create Train/Test Split datasets


### 2. Training
* Training Loop
  1. Dataset 
  2. Forward Pass
     * Function to predict next word.
  3. Loss Computation
     * Loss function penalizes model if it predicts next word incorrectly. 
  4. Backpropagation
     * Model learns to minimize loss during this step. 
  4. Updating Weights
     * Model weights are updated based on gradients with respect to the loss computations.


* Things to consider when finetuning (training) a model
1. Hyperparameters
   * learning rate
   * learning rate scheduler
   * weight decay
   * Batch size
2. Saving model checkpoints at regular intervals
   * Important to monitor model failures in training such as runtime errors, etc…
   * Resuming training from intermediate checkpoint can save time and help resolve errors faster and more efficiently. 
3. Evaluation on an evaluation dataset
  * Always do this at regular intervals.
  * This will help catch any under or overfitting. 
4. Compute requirements and hardware
  * Need access to adequate compute resources and hardware.
5. Tracking experiments 
  * Track outputs of model training with framworks such as:
    * Weights and Biases
    * Tensorboard
    * ..etc...
   

* **Minimal PyTorch Training Loop**
  * However, we must remember that the code below can be cumbersome and quite complex when finetuning on larger datasets. However, it is still important to know. 
```
for step, (inputs, labels) in enumerate(train_dataloader):
	# forward pass
	predictions = model(inputs)
	# loss computation
	loss = loss_function(predictions, labels)
	# Backpropagation
	loss.backward()
	# Updating weights and reset gradients
	optimizer.step()
	optimizer.zero_grad()
```

* Hugging Face provides training and inference at Scale which makes this much easier than having to write boilerplate code!
  * This is available via the Transformers Trainer API and Accelerate
     * 1) Trainer API
         * Plug in the model and datasets and call `trainer.train()`.
         * No need to write “boilerplate PyTorch loops”.
     * 2) Accelerate Trainer API 
     * Training and inference at scale made simple, efficent and adaptable.
     * Use this when you want more control over your PyTorch training loop but at same time want to avoid alot of boilerplate code.




### 3. Evaluation
1. **Human Evaluations are the GOLD STANDARD**
   * They provide the true way now to properly gauge the model capabilities. 
   * The best way to evaluate models is to use the chatbot arena on huggingface:
      * Link: https://lmarena.ai/?leaderboard
2. **Model Evaluations**
   * Using SOTA models such as GPT-4 to evaluate performance of other models. 
   * Evaluations can be skewed though towards models trained with data generated using the model using the evaluation. 


3. **LLM Benchmarks**
   * A very noisy proxy!
   * ONLY useful to gauge generic capabilities of models. 



4. **Task Specific Testing Suite**
   * Often you have to build your own testing suite using an open or closed source tool. 



