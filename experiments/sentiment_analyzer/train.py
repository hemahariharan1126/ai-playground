import os
import sys
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
import evaluate

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.logger import logger
from experiments.sentiment_analyzer.model import get_model

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train(epochs=1, batch_size=8):
    model_name = "distilbert-base-uncased"
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    logger.info(f"Loading tokenizer and dataset for {model_name}...")

    # 1. Load Dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    # 2. Preprocessing & Tokenization
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # For a quick run/demo, we can use a small subset
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

    # 3. Build Model
    model = get_model(num_labels=2)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./models/sentiment_analyzer_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs/sentiment_analyzer",
        report_to=["mlflow", "tensorboard"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # 6. Evaluation
    logger.info("Evaluating model...")
    results = trainer.evaluate()
    logger.info(f"Evaluation Results: {results}")

    # Save final model
    trainer.save_model("./models/sentiment_analyzer_final")
    logger.info("Model saved to ./models/sentiment_analyzer_final")

if __name__ == "__main__":
    train()
