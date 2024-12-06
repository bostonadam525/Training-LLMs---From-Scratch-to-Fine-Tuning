# Full Fine Tuning LLMs
* A repo devoted to full fine tuning of LLMs.


## To run code in the notebook `instruction fine tuning` follow these steps:
* Recommend running on GPU on runpod
1. Open Runpod instance
2. Choose RTX A6000 instance from “secure cloud tab”
3. Choose PyTorch 2.1.1 docker image 
4. Increase storage to 100 GB via “customize dep

* Next do this:
  * Open Terminal in Jupyter Lab
  * pip install -r requirements.txt file
  * log into weights and biases: `wandb login` 
    * This will prompt you for your API key via your profile. 
  * log into huggingface hub using the command `Huggingface-cli login`
  * This will prompt you to enter your token
    * Make sure it has “write access” to push models to hub
