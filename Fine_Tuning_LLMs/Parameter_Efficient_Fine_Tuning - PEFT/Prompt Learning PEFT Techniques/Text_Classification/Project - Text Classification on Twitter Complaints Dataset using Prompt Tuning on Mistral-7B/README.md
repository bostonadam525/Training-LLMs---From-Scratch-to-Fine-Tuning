# Project - Text Classification on Twitter Complaints Dataset using Prompt Tuning on Mistral-7B
* In this project we will perform text classification on a twitter complaints dataset using an LLM by leveraging the power of PEFT methods specifically "Prompt Tuning" which is a sub category of Prompt Learning PEFT techniques.
* LLM we are prompt tuning
  * We will perform prompt tuning on the `Mistral-7B` model.
  * Here is the model card: https://huggingface.co/mistralai/Mistral-7B-v0.1

* To run this project do the following below.
* I first suggest getting a GPU instance like Runpod. 

* Steps
  1. Open Runpod instance
  2. Choose RTX A6000 instance from “secure cloud tab”
  3. Choose PyTorch 2.1.1 docker image 
  4. increase storage to 100 GB via “customize dep

* Start up the Runpod jupyter lab instance, then do the following:
  1. Open Terminal in Jupyter Lab
  2. pip install -r requirements.txt file
  3. log into weights and biases: wandb login 
    * This will prompt you for your API key via your profile. 
  4. log into huggingface hub
  5. Huggingface-cli login
    * This will prompt you to enter your token
    * Make sure it has “write access” to push models to hub
