# SageMaker LLMs and Fine Tuning
* Fine tuning LLMs using a resource such as SageMaker requires intensive resource requirements.


## Inference Resource Requirements
* For inference, 1 billion parameters at 32-bit floating point (FP) precision requires 4 GB of VRAM.
* How do we calculate this?
  * `32-bit FP = 4 Bytes`
  * `4 Bytes x 10^9 (1B Params) = 4 GB VRAM`
* As an example, **Llama-3-70B** inference cost would be:
  * `70 x 4 = 280 GB of VRAM`

* If you used the `AWS p4d.24xlarge`:
  * `96 vCPUs`
  * `1237 GB`
  * `8 x A100GPUs = 320 GB`
  * `On-demand = 32.77 USD/hr`
  * `Per month cost = $23,594.4 USD per month`
 

## Full Fine-Tuning on AWS SageMaker
* 1. **ISSUE #1 - Training GPU usage:**
  * For 1 parameter in transfomer weights
      * `2 x Optimizer states`
      * `1 x Gradients`
      * `2 x Activations and temp memory`
  * `4 GB weights = 4 x 6 = 24 GB @ 32FP`
 

* 2. **OTHER ISSUES WITH FULL FINE-TUNING**
      * Large memory size
      * Maintaining and deploying multiple models
      * Catastrophic forgetting on fine-tuning which leads to POOR RESULTS on general task
        * Model will often perform poorly on new or unseen datasets and tasks it was NOT finetuned on.
       

## Parameter Efficient Fine Tuning on SageMaker
* Multiple techniques including:
    * SFT
    * RLHF
    * PPO, DPO
 

### SFT - Supervised Fine Tuning
* All or some of base weights of model are kept frozen.
* Either small number of layers are trained.
* Or new layers are trained and added later on.


## PEFT Techniques
1. Prompt Tuning
   * Prompt tuning params are added to the input embeddings at the attention layers.

2. LoRA - Low Rank Adaptation
   * Add LoRA weights to transformer weights. Only update/train LoRA.
  

### LoRA - Low Rank Adaptation
* Reparameterize by adding low rank decomposition matrices to different layers of the transformer.
* Subset of params to fine tune - mostly the later layers as they contain most of the combined knowledge.
* Microsoft research showed that a value rank between 4-10 often yields good performance.
* LoRA intution
  * We start with 2 matrices as we see below and perform cross product on them which results in Matrix `W` which is `D*N`.
  * Matrix W is equal to the size of the transformer layer and can be added to the transformer layer.
    * Also important, is that the final Matrix weights of W DOES NOT HAVE R IN IT.
    * This means that we can increase the value of R if we want more params to be trained, or less if we want less params to be trained.
    * There is a point of diminishing return.
* LoRA equation:

![image](https://github.com/user-attachments/assets/bbc70166-2cda-4380-93ab-b465a0b66dfb)


#### Advantages of LoRA
* Single GPU training for 7-10 billion parameter models ---> HUGE COST SAVINGS!!!!!!!!!!!
* **Scores are comparable to FULL FINE TUNED models**
* Many LoRA adapters can be trained for several downstream tasks with the SAME BASE MODEL.
* No added latency due to fused weights at runtime.


#### LoRA Workflow
* This is a typical LoRA workflow in SageMaker

1. Prepare dataset for fine-tuning.
2. Foundation Model as base model.
3. Fine tune using Hugging Face libraries.
   * Quantization used to reduce model size to fit into VRAM 
   * QLoRA based supervised fine-tuning (SFT) --> use quantized model to train the model weights. 
4. Evaluate the LLM outputs



# SageMaker Pipelines
* A pipeline is a series of steps with a specific purpose.
* A directed acyclic graph is automatically created (DAG).
* Automatically manages and creates EC2 infrastructure.
* Also available as Python SDK.
* Fully integrated with SageMaker Studio UI.

## Why do we even need a pipeline?
1. Automatic step retry Step Caching
2. Global Parameters
3. Data lineage tracking and integration with S3 buckets for IO
   * pipelines need data to be stored and feteched from S3.
4. Containerization
   * This forces us to containerize our data, results and workflow rather than keep them in a jupyter notebook.
  
5. Infrastructure provisioning support for spot-instances
6. SageMaker Experiments Integration
   * Automatically tracks hyperparameters, model artifacts, evaluation metadata, and other metrics.
  


## What are the pipeline steps that SageMaker Supports?

### 1. **Processing Step**
   * This is used to create a processing job specifically made for data processing.
  
  
#### Use Case:
      * For data pre and post processing
      * For model evaluation after training
      * Can also output files as evaluation reports
    
#### How Processing Works in pipelines
      * Sagemaker will start processing job with an EC2 machine
      * Takes the data from S3 bucket and copies it over EC2
      * Runs the container with given script
      * Copies the output artifacts back to S3
      * De-provisions the EC2 machine
    

### 2. **Training Step**
   * Used to create training job for training machine learning model(s).

#### Use Cases of training step
     * a. Train model with pre-built containers or use a custom container.
     * b. Hyperparameter tuning can run models in parallel to speed-up training time and hyperparameter search. 
     * c. Based on the tuning objective top 50 performing versions are retained.

#### How model training works
      * a. Sagemaker starts a job with EC2 machine(s).
      * b. Takes the training and validation data from S3 and copies it over to EC2.
      * c. Runs the container(s) with the given script.
      * d. Tracks container logs.
      * e. Tracks metrics using the container logs regex patterns --- specific patterns to track. 
      * f. After training de-provisions the EC2 machines. 


### 3. **Tuning Step**
   * Used to create a hyperparameter tuning job for a given model or models.


### 4. Model Inference Steps
  * Creates a new model for the model registry.
  * The model registry is a versioned system of managing models and has all the info needed to successfully retrieve/deploy the versioned model.

#### Transform Step
  * Used to batch run inference of a model on a given dataset.
  * These are the actual steps:
    1. Sagemaker starts a transform job with EC2 machines.
    2. Takes the data from S3 and optionally splits into chunks.
    3. Runs inference on the dataset.
    4. Optionally will aggregate the data and store the output data to S3.
    5. Afterwards de-provisions the EC2 machines. 
  
   
