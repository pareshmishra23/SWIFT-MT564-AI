"""
Data Preparation Script for TinyLlama Training
This script helps prepare data in the right format for TinyLlama training.
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for TinyLlama training")
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Path to input text files (accepts multiple files)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="training_data.json",
        help="Output JSON file with prepared data"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["instruction", "completion", "plain"],
        default="plain",
        help="Data format: instruction (instruction-response pairs), completion (text completion), or plain text"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Size of text chunks for plain text format"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks for plain text format"
    )
    return parser.parse_args()

def chunk_text(text, chunk_size, overlap):
    """Split text into overlapping chunks of specified size"""
    chunks = []
    start = 0
    
    # Skip empty or very short texts
    if len(text) < chunk_size / 2:
        return []
        
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Don't create tiny chunks at the end
        if end - start < chunk_size / 2 and chunks:
            # Extend the last chunk instead
            chunks[-1] = text[start - chunk_size + overlap:end]
            break
            
        chunks.append(text[start:end])
        start += chunk_size - overlap
        
    return chunks

def process_instruction_data(file_paths):
    """Process data formatted as instruction-response pairs"""
    data = []
    
    for file_path in file_paths:
        logger.info(f"Processing instruction data from: {file_path}")
        try:
            # Assuming JSON file with instruction-response pairs
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
            for item in tqdm(file_data):
                if "instruction" in item and "response" in item:
                    # Format as a prompt for TinyLlama
                    text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
                    data.append({"text": text})
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return data

def process_completion_data(file_paths):
    """Process data formatted for completion"""
    data = []
    
    for file_path in file_paths:
        logger.info(f"Processing completion data from: {file_path}")
        try:
            # Assuming JSON file with prompt-completion pairs
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
            for item in tqdm(file_data):
                if "prompt" in item and "completion" in item:
                    text = f"{item['prompt']}{item['completion']}"
                    data.append({"text": text})
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return data

def process_plain_text(file_paths, chunk_size, overlap):
    """Process plain text files by chunking them"""
    data = []
    
    for file_path in file_paths:
        logger.info(f"Processing plain text from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            chunks = chunk_text(text, chunk_size, overlap)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            for chunk in chunks:
                data.append({"text": chunk})
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return data

def main():
    args = parse_args()
    
    # Process data based on format
    if args.format == "instruction":
        data = process_instruction_data(args.input_files)
    elif args.format == "completion":
        data = process_completion_data(args.input_files)
    else:  # plain text
        data = process_plain_text(args.input_files, args.chunk_size, args.overlap)
    
    logger.info(f"Total processed examples: {len(data)}")
    
    # Save processed data
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Data saved to {args.output_file}")

if __name__ == "__main__":
    main()