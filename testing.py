from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Path to your fine-tuned model
model_path = r"C:\Users\vladc\OneDrive\Desktop\anonero.py\mistral-books-finetuned"
base_model_id = "mistralai/Mistral-7B-v0.1"

# Configure 4-bit quantization (same as training)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with same quantization settings
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load the LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, model_path)

print("✅ Model loaded successfully!")
print(f"Using device: {next(model.parameters()).device}")

# Create generation pipeline WITHOUT device parameter
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
    # Remove device=0 parameter since model is already on correct device
)

# Test prompt
prompt = "Explain the main theme of addiction and recovery in simple words."

# Generate output
print("Generating response...")
output = generator(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)

print("\n" + "="*50)
print("MODEL OUTPUT:")
print("="*50)
print(output[0]["generated_text"])




