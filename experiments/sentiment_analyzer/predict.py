from transformers import pipeline
import sys
import os

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.logger import logger

def predict(text, model_path="./models/sentiment_analyzer_final"):
    """
    Predicts sentiment of a given text.
    If fine-tuned model doesn't exist, falls back to base distilbert for demo.
    """
    model_to_use = model_path if os.path.exists(model_path) else "distilbert-base-uncased-finetuned-sst-2-english"
    
    if not os.path.exists(model_path):
        logger.warning(f"Fine-tuned model not found at {model_path}. Using default pre-trained model.")

    logger.info(f"Initializing sentiment-analysis pipeline with {model_to_use}...")
    classifier = pipeline("sentiment-analysis", model=model_to_use)
    
    result = classifier(text)[0]
    logger.info(f"Input: {text}")
    logger.info(f"Prediction: {result['label']} (Confidence: {result['score']:.4f})")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        predict(text)
    else:
        sample_text = "The AI Playground project architecture is fantastic and very easy to follow!"
        predict(sample_text)
