import os
import sys
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

# Add project root to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.logger import logger
from experiments.text_generator.model import get_model

def train(epochs=1, batch_size=4):
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading tokenizer and dataset for {model_name}...")

    # 1. Load Dataset (using a creative/literary dataset)
    # Tiny Shakespeare is a classic for creative generation tests
    dataset = load_dataset("tiny_shakespeare")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 doesn't have a padding token by default
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    # 2. Preprocessing & Tokenization
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    # 3. Build Model
    model = get_model(model_name)
    model.to(device)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./models/text_generator_checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir="./logs/text_generator",
        report_to=["mlflow", "tensorboard"],
        evaluation_strategy="no"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False # GPT-2 is Causal LM, not Masked LM
    )

    # 5. Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model("./models/text_generator_final")
    logger.info("Model saved to ./models/text_generator_final")

if __name__ == "__main__":
    train()
