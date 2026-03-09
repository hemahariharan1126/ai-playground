from transformers import GPT2LMHeadModel, GPT2Config

def get_model(model_name="gpt2"):
    """
    Initializes a GPT-2 model for causal language modeling.
    """
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    return model
