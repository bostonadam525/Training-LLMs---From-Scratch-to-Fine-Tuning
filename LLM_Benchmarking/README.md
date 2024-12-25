# LLM Benchmarking
* A repo dedicated to all things LLM benchmarking.


# Live Bench
* This is the original paper: https://arxiv.org/abs/2406.19314
* Livebench Leaderboard: [https://livebench.ai/#/](https://livebench.ai/#/)
* Livebench github: https://github.com/livebench/livebench

# Benchmarks for LLMs
1. **GPQA (Google Proof Q+A)**
  * Tests an LLM ability to answer questions that are graduate level in subjects such as biology, physics, chemistry. 

2. **MMLU (Massive Multitask Language Understanding)**
  * Testing across 57 different multilingual subjects such as:
      * Law
      * Medicine
      * Math
      * History
  * Utilized to determine knowledge and reasoning in specific domains. 

3. **HumanEval**
  * Python programming problems that test an LLMs ability to complete functions. 
  * LLM models must write code that has to pass unit tests. 

4. **MATH**
  * Testing advanced mathematical problem solving at high school “Olympiad level”.
  * Math problems require multi-step reasoning and understanding. 

5. **MGSM (multilingual grade school math)**
  * Tests both linguistic comprehension and mathematical reasoning. 
  * Problems are “grade school level” but requires task of translation. 

6. **DROP (Discrete Reasoniing Over Paragraphs)**
  * Testing reading comprehension that requires numerical reasoning and manipulation. 
  * LLMs are tested on their ability to perform math calculations from text. 

# Problems with Benchmarks!!
1. **Real World Data**
  * These benchmarks may or often may not actually represent real world interactions and real world data and their unique challenges they present to LLMs. 
  * Some of the eval benchmarks above are often considered too “simple” and “easily measurable”. 
  * The more difficult to measure metrics and capabilities of LLMs are often ignored by these benchmarks. 
  * Overall, these benchmarks are exactly that, benchmarks, and translation to real-world uses cases is not a 1:1 scenario. 


2. **Saturation & Overfitting**
  * Benchmark models are already becoming “saturated and overfitted” meaning they are **TOO OPTIMIZED** for what they are testing and not representative enough of real world challenges. 
  * Prevents ability to test LLMs ability to generalize on unseen data.



# New LLM Benchmarks
## 1. **SWE-Bench (software engineering benchmark)**
  * Able to evaluate how well AI models and LLMs can identify and fix software engineering bugs in real projects. 
  * Tests if LLMs can do things similar to a human such as:
      * Understand codebases
      * Interpret bug reports
      * Write correct patches
  * Benchmark contains ~50,000 historical bug fixes from popular open source Python projects. 
  * Using these benchmarks, an LLMs “real world coding” capabilities are tested such as:
      * Code modifications without regressions
      * Test maintenance 
      * Bug fixing
      * Understanding software engineering codebases


### How this works:
  * An LLM will be given 3 key inputs:
      * Entire codebase frozen at specific commit version
      * Bug report or issue describing the problem(s) the LLM needs to fix. 
      * Comprehensive Test suite to validate if bug fixes work
  * The LLM then must generate a working solution.
  * “Success” is judged by 2 metrics:
      * **Fail-to-Pass** —> reveals if the model has actually fixed the failing tests. 
      * **Pass-to-Pass** —> reveals if the model avoided breaking any existing code that still works. 

### Why this works
  * Real world use cases rather than “theoretical”. 
  * Ensures morel longterm relevance to enhance LLM abilities. 




## 2. T-bench (TAU-bench)
  * Benchmark designed to evaluate LLMs ability to automate tasks in various scenarios. 
  * Most importantly, this benchmark focuses on testing a model’s ability to handle real-world business applications and data structures. 
  * Simulates interacting with databases and real world systems.


### Why this is unique?
  * Focuses on measuring success over a series of tasks using realistic examples evaluating end to end workflows interacting with databases and APIs. 
  * There are 2 domains currently supported:
      * Retail
      * Airline
   


# How to pick the right LLM model 
* Reference from Anthropics guidelines


## Three Key Metrics to consider
1. Accuracy / capabilities
2. Speed
3. Cost


### Capabilities (Accuracy)
* This does not just focus on machine learning or deep learning test set metrics, but also features and performance. 
* First question you need to ask: Does the LLM you have picked have the necessary capabilities to handle tasks and use cases that are specific to your domain and application? 
* Generally speaking:
  * All LLMs have various levels of performance across different domains such as:
      * General Language Understanding
      * Task-specific knowledge
      * Reasoning abilities
      * Generation quality
* It is paramount to align the LLM model strengths with the demands of the application you are building —> this will ensure optimal results. 
* Accuracy is the most common metric used, but remember that accuracy is only a measure of how close the output results are to the True, known or expected value. 
* Precision would be another metric, as it compares results to one another. 


### Speed
* The speed of which an LLM can intake a user query and output a generative AI response is crucial in the model you select. 
* This is especially important in real-time interactions.
* Generally speaking:
  * Faster LLM models can provide a stress free user experience reducing important measures such as LATENCY and thus improving usability. 
  * Balance between SPEED and CAPABILITIES is important because a faster model may not always give you the best accuracy or results. 


### Cost
* LLMs with more features and parameters often have a higher price tag if they are closed-source or proprietary.
  * For every API call the cost adds up quickly. 
* Therefore it is paramount to assess the cost implications of your application and determine what is cost-efficient and effective. 
