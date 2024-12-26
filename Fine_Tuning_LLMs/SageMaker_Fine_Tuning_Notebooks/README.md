# SageMaker Fine Tuning
* This repo walks through fine tuning LLMs using AWS SageMaker.


## SageMaker Instances
* Full SageMaker AI pricing list: https://aws.amazon.com/sagemaker-ai/pricing/


1. Standard instances
   * t series
   * m series

2. Compute optimized
   * c series
  

3. Memory-optimized
   * r series (RAM optimized, e.g. heavy analytics workflows)


4. GPU optimized
   * g series
   * p series
  

5. Deep Learning Inference Optimized
   * inf series


## Training with SageMaker
* There are **3 basic options**

1. **Built-in Algorithms**
  * These provide support for common algorithms from various providers.
  * These work "out of the box" or immediately. 
  * Default support for common frameworks such as:
    * PyTorch
    * Tensorflow
    * Hugging Face
    * Chainer
    * Scikit-learn
    * MXNet
    * Spark ML
   
2. **Script Mode**
   * Container provided by AWS.
   * A use has to provide a **script file** to run with the container.
   * Allows support for BOTH **training and deployment**
  

3. **Custom Container**
   * Allows you to create your own deep learning container.
       * Can run your own docker container in SageMaker as well. 
   * Support for training and deployment with other SageMaker resources.
  


## SageMaker Studio for Training
* Typical Machine Learning Workflow in SageMaker is as follows in the diagram below:

![image](https://github.com/user-attachments/assets/0c233dd3-4960-4e8d-bb89-15e65f300b06)



### What do we actually need to track in an ML Workflow?
1. Dataset
   * The dataset obviously goes through many changes overtime.
   * We need to track the Data Preprocessing code over time.
  
2. Build
   * Various versioned libraries and frameworks change over time and we need to track these so they are reproducible.
  

3. Train
   * **Hyperparameters** of machine learning/deep learning models need to be tracked.
   * **Experiment Logs** - tells us at exact points what happened.
   * **Training Metrics**
  
4. Evaluate
   * **Evaluation Metrics** -- these are after the model has been trained and inference is run.
   * **Model Weights** -- if the output was satisfactory, we save them as model **artifacts**.
   * **Compare different ML model runs together** -- choose the best model
  

## SageMaker Integrations
* REST APIs
* AWS CLI
* SDK

### Python SDK (pypi)
* `sagemaker` package
* `boto3`package --> talk to AWS via SDK

### Other SageMaker Services
1. SageMaker JumpStart
   * Allows you to quickly test Foundation Models.
   * Test end-to-end solutions from AWS Marketplace.

2. Bedrock
   * Pay as you go Foundation Models in various AWS regions.
   * Access to hosted LLMs.


## SageMaker Experiments
* This is an excellent diagram showing how SageMaker Experiments flow:
  * `Development Environment` is a Jupyter Lab Instance where we write our code.
  * `SageMaker Experiments` -- we can log onto the SageMaker UI and see all outputs from our experiments. This is a "central hub" of all results.
  * `Model Registry` --- area to save our best model(s).

![image](https://github.com/user-attachments/assets/fa14a1d3-0c55-4227-8411-3eac6f58141c)


* SageMaker Experiments usually consist of:
  1. Multiple runs
  2. Multiple metrics logged and stored in `SageMaker Experiment`
  3. Deep integration with AWS SDK
  4. Organized in Studio dashboard user interface. 

