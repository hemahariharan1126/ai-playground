from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_model(model_name="microsoft/DialoGPT-medium"):
    """
    Initializes a DialoGPT model for conversational AI.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # DialoGPT doesn't have a padding token, using EOS
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
