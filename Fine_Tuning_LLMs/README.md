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


## Breakdown of the Finetuning Process
* Data quality is the KEY during finetuning.
* Here are the primary steps:
 * 1) Data gathering and Preprocessing
 * 2) Training
 * 3) Evaluation
* This process is iterative and repeated multiple times with finetuning the parameters and new data.


### Pre-Training vs. Finetuning
* This is again a very common debate and topic of discussion.
* A major difference to remember:
   * Pre-Training uses self-supervised learning — there is NO annotated or labeled data. 
   * Finetuning uses Supervised learning WITH annotated and labeled data which is highly recommended. 
Self-supervised learning is also supported.
* Here is an excellent comparison of the 2 methods (source: Analytics Vidhya):
![image](https://github.com/user-attachments/assets/7279f696-7013-4f1a-a54e-8be0d434b648)




