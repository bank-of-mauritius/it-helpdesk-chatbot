#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a sequence-to-sequence model for IT helpdesk responses using Hugging Face's Transformers.
"""

import json
import argparse
import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class HelpDeskDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []

        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    self.inputs.append(item['prompt'])
                    self.targets.append(item['response'])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line}")
                except KeyError as e:
                    logger.warning(f"Missing key in JSON: {e}")

        logger.info(f"Loaded {len(self.inputs)} examples")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encoding = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            self.targets[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token id with -100 so it's ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_model(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Initialize tokenizer and model
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # Make sure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset
    dataset = HelpDeskDataset(args.data_path, tokenizer, max_length=args.max_length)

    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete")

def main():
    parser = argparse.ArgumentParser(description="Train a helpdesk chatbot model")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Name of the pre-trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./helpdesk_bot_model", help="Output directory for the model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of saved checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()