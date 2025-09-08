import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

# Config
model_id = "mistralai/Mistral-7B-v0.1"
dataset_file = "books_chunked.jsonl"
output_dir = "./mistral-books-finetuned"

# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_file, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = dataset["train"], dataset["test"]

print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Tokenize without padding yet
    return tokenizer(examples["text"], truncation=True, max_length=1024)

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "file"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text", "file"])

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model with 4bit quantization
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Training arguments - updated for newer Transformers version
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    eval_strategy="steps",  # Changed from evaluation_strategy
    save_strategy="steps",
    warmup_steps=100,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train
print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model fine-tuned and saved to {output_dir}")
