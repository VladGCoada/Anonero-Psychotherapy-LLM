from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Your existing loading code...
model_path = r"C:\Users\vladc\OneDrive\Desktop\anonero.py\mistral-books-finetuned"
base_model_id = "mistralai/Mistral-7B-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, model_path)

# Get the device the model is on
device = next(model.parameters()).device
print(f"Model is on device: {device}")

# Test with simple prompts to understand model behavior
test_prompts = [
    # Very simple statements
    "Hello,",
    "The sky is",
    "Addiction means",
    "Recovery involves",
    
    # Direct questions
    "What is addiction?",
    "Define recovery.",
    "Tell me about addiction.",
    
    # Instructional prompts
    "Explain: addiction",
    "Describe: recovery",
    "Summary: addiction and recovery",
]

print("Testing model behavior with simple prompts:\n")
print("=" * 80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Prompt: '{prompt}'")
    print("-" * 40)
    
    # Tokenize and move to the same device as model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to same device
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Show only the new generated part (after the prompt)
    if response.startswith(prompt):
        generated_part = response[len(prompt):].strip()
        print(f"Generated: '{generated_part}'")
    else:
        print(f"Full response: '{response}'")
    
    print("-" * 40)

# Test with directive prompts
directive_prompts = [
    "Answer the following question: What is addiction?",
    "Provide a definition for: recovery",
    "Give a straightforward explanation of addiction:",
    "Write a paragraph about recovery:",
]

print("\n\nTesting with directive prompts:")
print("=" * 80)

for i, prompt in enumerate(directive_prompts, 1):
    print(f"\n{i}. Prompt: '{prompt}'")
    print("-" * 40)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if response.startswith(prompt):
        generated_part = response[len(prompt):].strip()
        print(f"Response: {generated_part}")
    else:
        print(f"Full response: {response}")
    
    print("-" * 40)
    