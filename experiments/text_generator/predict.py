from transformers import pipeline, AutoTokenizer
import sys
import os

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.logger import logger

def generate_text(prompt, model_path="./models/text_generator_final", max_length=100):
    """
    Generates text based on a prompt.
    Falls back to base GPT-2 if fine-tuned model isn't found.
    """
    model_to_use = model_path if os.path.exists(model_path) else "gpt2"
    
    if not os.path.exists(model_path):
        logger.warning(f"Fine-tuned model not found at {model_path}. Using base GPT-2.")

    logger.info(f"Initializing text-generation pipeline with {model_to_use}...")
    
    # Initialize generator
    generator = pipeline(
        "text-generation", 
        model=model_to_use, 
        tokenizer="gpt2" if model_to_use == "gpt2" else model_to_use
    )
    
    logger.info(f"Prompt: {prompt}")
    
    # Generate
    results = generator(
        prompt, 
        max_length=max_length, 
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    
    generated_text = results[0]['generated_text']
    logger.info(f"Generated Text:\n{generated_text}")
    
    return generated_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        generate_text(prompt)
    else:
        sample_prompt = "The future of AI is"
        generate_text(sample_prompt)
