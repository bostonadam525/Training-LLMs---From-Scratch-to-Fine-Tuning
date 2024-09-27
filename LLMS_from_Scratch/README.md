# LLMS from Scratch
* Repo by Adam Lang
* A repo devoted to all things related to building and training LLMs from scratch.

## Steps Involved
1. Training Data curation
   * Data includes open web data, your data, or other corpora
   * Principles of training data for LLMs
        1. Massive Scale (e.g. billions of tokens)
        2. Diversity of datasets — cross domain knowledge 
        3. High quality datasets!!!!!
           * Small language models like Phi-3 work on higher quality datasets
   * Open source datasets
        1. Common Crawl —> AWS 
        2. Refined Web Dataset —> falcon LLM training (huggingface)
        3. Pile —> Academic, Internet, Prose, Dialogue, Misc
        4. The Stack —> programming languages 
        5. Math Pile —-> huggingface —> math specific 
   * Steps involved in curating your training data
        1. Estimate training data size using scaling laws
        2. Focus on high-quality training data
2. Data Preprocessing
   * High quality training data —> powerful models
   * Thus, data cleaning is paramount to success! 
   * Filtering out raw data to create high-quality training dataset.
   * Pre-processing steps:
        1. Sampling datasets to handle training data distribution — over sample or undersample from various sources
           * Low quality datasets tend to be under sampled
           * High quality datasets tend to be over sampled
           * As an example, GPT-3 training featured “over-sampling” of higher quality datasets such as Wikipedia (3.4), WebText2 (2.9), Books1 (1.9), etc..
        2. Data Deduplication
           * Remove duplicate text across training data!!
           * Why data deduplication?
                1. Efficient model training — eliminate memorization of the training data
                2. Accelerate training process (faster training! — training set sizes reduced)
                3. Accurate evaluation — reduce train/test overlap
           * Methods for data deduplication
                1. Jaccard Similarity b/w document pair-wise
                   * Intersection / union of bigrams
                   * Not scalable for LARGE datasets!!! Pairwise comparisons
                2. MinHash
                   * Novel technique to compute jaccard similarity based on Hashing
                   * MinHash has 4 steps for data deduplication:
                        1. Tokenization (usually n-grams)
                        2. Fingerprinting — map each document into a set of hash values
                        3. Apply Locality-Sensitive Hashing (LSH) — reduce number of hash values place data into different “buckets”
        3. Removing Boilerplate text
        4. Eliminating HTML code
        5. Removing bias/harmful content
3. Tokenization
  * <to be continued>
4. Model architecture —> training
5. Model Evaluation

