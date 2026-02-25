import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch

# Small model that can actually train on free GPUs
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "./DataSet/data.json"
OUT_DIR = "./model_adapter"

def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your format: [{"inputs": "<s>### Instruction: ... ### Response: ...</s>"}]
    texts = [row["inputs"] for row in data if "inputs" in row and isinstance(row["inputs"], str)]
    return Dataset.from_dict({"text": texts})

def main():
    ds = load_dataset()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized = ds.map(tokenize, batched=False, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA adapter (this DOES NOT fine-tune the whole model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # common for LLaMA-like models
    )
    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=20,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ Saved LoRA adapter to: {OUT_DIR}")

if __name__ == "__main__":
    main()