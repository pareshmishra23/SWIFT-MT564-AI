"""
MT564 TinyLlama Training Script
This script fine-tunes a TinyLlama model on MT564 format specifications data.

we are using the model model_name = "sshleifer/tiny-gpt2"  # Or "distilgpt2"
for better meory
"""

import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama on MT564 format specifications")
    parser.add_argument(
        "--model_name",
        type=str,
        #default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        default="sshleifer/tiny-gpt2",
    help="Hugging Face model ID or path to local model"
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default="mt564_training_data.json",
        help="Path to the MT564 training data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        #default="./mt564_tinyllama_model",
        default="sshleifer/tiny-gpt2",
    help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of update steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use fp16 16-bit (mixed) precision training"
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for evaluation"
    )
    return parser.parse_args()

def format_training_data(data):
    """Format training data for the model"""
    formatted_data = []
    
    for item in data:
        # Format as a chat-like conversation for TinyLlama
        formatted_text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
        formatted_data.append({"text": formatted_text})
    
    return formatted_data

def main():
    args = parse_args()
    
    try:
        # Import necessary libraries
        import torch
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer,
            Trainer, 
            TrainingArguments,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
    except ImportError:
        logger.error("Required libraries not installed. Please install torch, transformers, and datasets.")
        return
    
    # Load training data
    logger.info(f"Loading training data from {args.training_data}")
    with open(args.training_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = format_training_data(data)
    logger.info(f"Formatted {len(formatted_data)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Split into training and evaluation sets
    dataset = dataset.train_test_split(test_size=args.eval_ratio)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )
    
    logger.info("Tokenizing datasets")
    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_eval = dataset["test"].map(tokenize_function, batched=True)
    
    # Create label column for causal language modeling
    tokenized_train = tokenized_train.map(
        lambda examples: {"labels": examples["input_ids"]},
        batched=True
    )
    tokenized_eval = tokenized_eval.map(
        lambda examples: {"labels": examples["input_ids"]},
        batched=True
    )
    
    # Remove the text column as it's no longer needed
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_eval = tokenized_eval.remove_columns(["text"])
    
    # Load model
    logger.info(f"Loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32
    )
    # to save MAC memory
    import torch
    device = torch.device("cpu")
    model.to(device)

    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to="none"
    )
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting fine-tuning")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()