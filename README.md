# SWIFT MT564 AI
# 🚀 SWIFT-MT564-AI: Anomaly Detection in Corporate Action Messages

This project focuses on detecting **structural and compliance anomalies** in SWIFT MT564 corporate action messages using a **fine-tuned Google Gemma 3B model** with LoRA (Low-Rank Adaptation).

---

## 💡 Objective

To build an AI assistant that:
- Parses SWIFT MT564 message structure
- Flags field-level and sequence-level anomalies
- Provides actionable insights for compliance and reconciliation teams

---

## 🔍 Model Details

- **Base Model**: `google/gemma-3-1b-it`
- **Fine-Tuned Using**: LoRA on `transformers + peft`
- **Training Format**: Instruction-tuned JSONL
- **Sample Prompt Format**:

Train TinyLlama or Gemma models to detect anomalies in SWIFT MT564 messages.

Instruction:
Analyze this MT564 message for anomalies

Input:
{1:F01BANKXXXX...}{2:I564BANKYYYY...}{4:
:16R:GENL
:20C::CORP//CA20250501
...
}

Response:
Missing Sequence B

Unusual currency

Sanctioned recipient

yaml
Copy
Edit

---

## 📁 Directory Structure

SWIFT-MT564-AI/
│
├── app.py # Inference interface (API or CLI)
├── main.py # Entry point
├── prepare_mt564_data.py # Training data generator
├── prepare_data.py # Generic data preprocessing
├── train_mt564_model.py # Trainer for baseline models
├── train_tinyllama.py # Legacy: TinyLlama training
├── requirements.txt # All dependencies
├── mt564-gemma-lora/ # Fine-tuned model dir
├── data/ # Raw training data (SWIFT JSON)
├── README.md # Project readme
├── .gitignore # Git exclusions

yaml
Copy
Edit

---

## 🧪 Local Inference

```bash
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("pareshmishra/mt564-gemma-lora")
model = AutoModelForCausalLM.from_pretrained("pareshmishra/mt564-gemma-lora")

input_text = "### Instruction:\nAnalyze this MT564 message for anomalies\n\n### Input:\n{1:F01...}"
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
🔄 Training (LoRA)
See train_mt564_model.py or use Colab/AutoTrain with the formatted JSONL file.

🧠 Model Hosted at
👉 Hugging Face Model Page

📚 Resources Used
SWIFT ISO20022 documentation

Custom JSONL training entries

Google Gemma 3B IT Model

Hugging Face transformers, peft, datasets

📬 Contact
For questions or contributions: @pareshmishra

MIT License. For research or educational use only.
 






 

