## Standard DS imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from datetime import datetime

## ML imports
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, ## text generation head
    BitsAndBytesConfig, ## we need 4-bit config
    TrainingArguments, ##fine tuning
    Trainer, ## fine tuning
    DataCollatorForLanguageModeling, ## fine-tuning
    pipeline, ## for testing basemodel and inference
)
from datasets import load_dataset, Dataset, DatasetDict ## HF dataset, dataloaders
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model ## LoRA config for fine-tuning
from trl import SFTTrainer ## reinforcement learning from supervised fine-tuning
from accelerate import Accelerator
