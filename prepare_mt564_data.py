"""
MT564 Data Preparation Script
This script converts MT564 format specifications into training data for TinyLlama
"""

import os
import json
import argparse
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MT564 format specifications for TinyLlama training")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the MT564 format specification JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="mt564_training_data.json",
        help="Output file to save the prepared training data"
    )
    return parser.parse_args()

def create_instruction_examples(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert MT564 specification into instruction-response pairs"""
    examples = []
    
    # Extract sequences from the specification
    #sequences = data.get("MT564FormatSpecifications", {}).get("sequences", [])
    sequences = data  # if the input is already a list

    # Create examples for sequence overview
    sequence_names = [seq["name"] for seq in sequences]
    examples.append({
        "instruction": "What are the main sequences in the MT564 message format?",
        "response": f"The MT564 message format consists of the following main sequences:\n\n" + 
                    "\n".join([f"- {name}" for name in sequence_names])
    })
    
    # Create examples for each sequence
    for sequence in sequences:
        seq_name = sequence["name"]
        status = sequence["status"]
        
        # Example for sequence details
        examples.append({
            "instruction": f"Describe the '{seq_name}' sequence in MT564 format.",
            "response": f"The '{seq_name}' sequence is {status}. " +
                        f"It contains {len(sequence.get('fields', []))} fields " +
                        f"and {len(sequence.get('subsequences', []))} subsequences."
        })
        
        # Examples for fields in the sequence
        fields = sequence.get("fields", [])
        
        field_examples = []
        for field in fields:
            field_tag = field.get("tag", "")
            field_name = field.get("fieldName", "")
            field_status = field.get("status", "")
            field_content = field.get("content", "")
            
            field_examples.append(f"- {field_tag}: {field_name} ({field_status}) - Format: {field_content}")
        
        if field_examples:
            examples.append({
                "instruction": f"List the fields in the '{seq_name}' sequence of MT564.",
                "response": "The fields in this sequence are:\n\n" + "\n".join(field_examples)
            })
        
        # Examples for subsequences
        subsequences = sequence.get("subsequences", [])
        if subsequences:
            subseq_names = [subseq["name"] for subseq in subsequences]
            examples.append({
                "instruction": f"What subsequences are part of the '{seq_name}' sequence in MT564?",
                "response": f"The '{seq_name}' sequence contains the following subsequences:\n\n" + 
                            "\n".join([f"- {name}" for name in subseq_names])
            })
            
            # Create examples for each subsequence
            for subsequence in subsequences:
                subseq_name = subsequence["name"]
                subseq_status = subsequence["status"]
                subseq_fields = subsequence.get("fields", [])
                
                examples.append({
                    "instruction": f"Describe the '{subseq_name}' subsequence in MT564 format.",
                    "response": f"The '{subseq_name}' subsequence is {subseq_status}. " +
                                f"It contains {len(subseq_fields)} fields."
                })
                
                # Examples for fields in subsequence
                subfield_examples = []
                for field in subseq_fields:
                    field_tag = field.get("tag", "")
                    field_name = field.get("fieldName", "")
                    field_status = field.get("status", "")
                    field_content = field.get("content", "")
                    
                    subfield_examples.append(f"- {field_tag}: {field_name} ({field_status}) - Format: {field_content}")
                
                if subfield_examples:
                    examples.append({
                        "instruction": f"List the fields in the '{subseq_name}' subsequence of MT564.",
                        "response": "The fields in this subsequence are:\n\n" + "\n".join(subfield_examples)
                    })
    
    # Add generic examples
    examples.append({
        "instruction": "What is an MT564 message used for?",
        "response": "The MT564 is a SWIFT message format used for Corporate Action Notifications. It is sent by an account servicer to an account owner to notify them about a forthcoming corporate action event and the options or choices available to the account owner. This may include events such as dividend announcements, rights issues, stock splits, mergers, and other corporate events that affect securities."
    })
    
    examples.append({
        "instruction": "Explain the structure of MT564 messages.",
        "response": "MT564 messages follow a structured format with sequences and subsequences:\n\n" +
                   "1. Sequence A - General Information (Mandatory): Contains basic information about the message\n" +
                   "2. Sequence B - Underlying Securities (Mandatory): Details about the affected securities\n" +
                   "3. Sequence C - Intermediate Securities (Optional): Information about intermediate securities\n" +
                   "4. Sequence D - Corporate Action Details (Optional): Details about the corporate action\n" +
                   "5. Sequence E - Corporate Action Options (Optional): Available options for the account owner\n" +
                   "6. Sequence F - Additional Information (Optional): Any additional relevant information\n\n" +
                   "Each sequence contains specific fields, identified by tags, that carry different pieces of information."
    })
    
    return examples

def main():
    args = parse_args()
    
    # Load MT564 format specification
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create instruction-response pairs
    examples = create_instruction_examples(data)
    
    print(f"Created {len(examples)} training examples from MT564 specifications")
    
    # Save the prepared data
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved training data to {args.output_file}")

if __name__ == "__main__":
    main()