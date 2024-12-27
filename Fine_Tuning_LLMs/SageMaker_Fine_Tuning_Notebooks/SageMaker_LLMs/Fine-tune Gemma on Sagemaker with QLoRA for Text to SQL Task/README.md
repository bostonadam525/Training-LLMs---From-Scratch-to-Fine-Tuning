# Fine-tune Gemma on Sagemaker with QLoRA
* In this project I demonstrate finetuning Gemma-2b LLM on SageMaker with QLoRA for Text to SQL Task.
* Fine Tuning is done using the "Script" Method to deploy the model for a SageMaker training job and then deployment to a SageMaker endpoint.
* Dataset is uploaded to S3 bucket and S3 utilized during process for fetching model parameters and artifacts. 

## Workflow Overview
1. Notebook running with a different EC2 instance type on AWS SageMaker
2. Use Huggingface library for:
   * dataset download - SQL generator Dataset from Huggingface
   * Transformer model - Gemma-2b
3. SFT Trainer from TRL library by Hugging Face
4. QLoRA based training from PEFT
5. Deploy using saved artifacts on S3
6. Evaluate the model
