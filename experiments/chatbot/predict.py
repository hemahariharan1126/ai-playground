import torch
import os
import sys

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.chatbot.model import get_model
from utils.logger import logger

def chat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Initializing Chatbot (DialoGPT-medium)...")
    model, tokenizer = get_model()
    model.to(device)
    
    chat_history_ids = None
    
    print("JARVIS: Online and ready to assist. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("JARVIS: Powering down. Connectivity terminated.")
            break
            
        # Encode user input and add EOS token
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
        
        # Append new user input to chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
        
        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        # Decode and print response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"JARVIS: {response}")

if __name__ == "__main__":
    chat()
