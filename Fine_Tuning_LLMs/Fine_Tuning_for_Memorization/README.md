# Fine-Tuning LLMs for Memorization

* Fine-tuning a large language model (LLM) for memorization is done to specifically enhance its ability to accurately recall and retrieve specific information from a targeted dataset, particularly when dealing with specialized knowledge or facts that might not be well represented in the LLM's general training data, allowing it to perform better on tasks requiring precise information retrieval within a specific domain.

## Why would you do this?
1. Custom or domain specific knowledge base
2. Improved accuracy in specific tasks
3. Data-driven approach
* Generally speaking:
  * if you extract text from a document —> it is difficult to have a model “memorize” the data
  * This is why you need various “angles” of the data for an LLM to have complete statistical view of your data.

## Recent paper on memorization fine tuning
* [Exploring Memorization in Fine-tuned Language Models](https://arxiv.org/html/2310.06714v2)
* Findings:
  * The authors concluded that memorization might be closely related to the information needed to complete a certain language task.
  * Language tasks such as sentiment analysis or extractive QA, only a few words or sentences are enough for the model to complete the task. 
    * For example, you can determine the sentiment based on some specific words in the sentiment, and can answer a question based on certain pieces of information.
    * In this case, the model only needs to learn **specific key features and is less likely to memorize the other data.**
  * However, for tasks such as **summarization and dialog**, this requires the model to learn more input features to complete the task, as the essential information from these inputs is also reflected in the output.
    * As a result, the fine-tuning process will encode more input knowledge from the data in the model parameters, leading to potential concerns of memorization.

## Domain examples where this technique excels
1. Medical chatbots
2. Finance
3. Legal research

## Considerations
1. Data quality is very important!
2. Memorization vs. Generalization
  * Remember: You Do not want to overfit on memorization as with any machine learinng model overfitting is an issue!!! 


# Reversal Curse
* original paper on this: [The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"](https://arxiv.org/abs/2309.12288)

* The “Reversal Curse” refers to the inability of LLMs to reverse causal statements they are trained on.
    * For instance, if an LLM is trained on a statement like “George Washington was the first US president,” it struggles to deduce the reverse, answering questions like “Who was the first US president?”

* The training dataset (e.g. website data crawl) is one of the main reasons we see this. Very rarely is the “reverse” of common terms and phrases present in the training data. 
    * This means that if you have a very concise document and the order of terminology is usually only in 1 order, an LLM usually will not be able to “memorize” all of the statisical probabilities of the token orders. 
* Given an input sentence can an LLM learn all possibly semantic probable ways the text is used? 

## Example sentence:
* "If the ball is pushed forward, the defending team is given a penalty."

* But, you need other iterations for the model to learn such as this:
  * "Defending team gets a penalty if the ball is pushed forward."
  * "Whats the result if the ball is passed foward? A penalty."

* So, given an input dataset how can we create an accurate dataset? 



# How to build a synthetic Q&A data set for LLM memorization fine-tuning?
* As an example:
  * Content: NFL football
  * Text: "If a player catches the ball in the end zone it’s a touchdown."

* Given the text above:
  * Ask an LLM to create a nuanced question and answer dataset. 
  * The question must include the context. 


## Data Expansion
1. tokenize and chunk a dataset
2. Ask an LLM to generate a set number of questions per chunk of text (e.g. 5 questions per chunk)
  * e.g. 1 question per 60 tokens

### First approach:
* ask different questions
  * e.g. Simple vs. Detailed questions and answers 

### Second approach:
* send the SAME request to the LLM but with various Temperature settings (e.g. deterministic vs. probabilistic) to get different outputs. 


# Fine-Tuning Setup
* Create a synthetic Q&A dataset or synthetic dataset based on your use case.
* Generally speaking, create 1 question per 60 tokens
* Expanded it 9 times at temperatures evenly spaced from 0.01 up to 1.2. Expanding dataset by factor of 9 by repeating same request 9 times. 
* Model: your choice!



# Batching for LLM fine tuning
* Usually we feed 1 row at a time
* Batching allows training in parallel
* GPUs are very good at parallel processing
* Losses from each row is aggregated for backward pass


## Batch size issues for memorization fine tuning

### Batch size of 1
* 1 row of data forward and backward pass
* model weights at 0.1
  * do the forward pass
  * calculate loss
* backpropagation loss through network and end up at weights of 0.2
* Do it again with weights at 0.2
  * do the forward pass with row 2 (batch 2)
  * calculate the loss
* backpropagation loss through network end up with weights at 0.3

### Batch size of 2
* With a larger batch size, we can run a parallel calculation model weights and losses of 2 rows at same time —> sum losses of both
  * So we get: Back prop sum of loss A + B
* Increasing the batch size is averaging all rows that are batched which should result in less noisy data
* However, this can lead to information loss!!!
* Generally speaking, memorization does better with lower batch sizes so you get specific details


## Benefits of smaller batch sizes
1. More granular learning/fitting of training data
2. Less VRAM usage

## Disadvantages of smaller batch sizes
1. Slower training
2. Overfitting the model


# Choosing Learning Rates
1. Best practice? Start at 1e-4 and Increase until training and validation loss oscillates too much
2. Most ideal you want the validation loss to be “smooth”
  * "not jumping up and down"


# Number of Epochs?
1. Train at “CONSTANT” learning rates
2. Increase Epochs until validation loss goes up

  * Example: if eval loss goes up after 2 epochs, re-train model for 2 epochs (optionally using cosine or linear)


# Model choice?
* start with model that does well “un-tuned” on sample questions





