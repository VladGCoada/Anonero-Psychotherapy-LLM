from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load your fine-tuned model
model_path = r"C:\Users\vladc\OneDrive\Desktop\anonero.py\mistral-books-finetuned"
base_model_id = "mistralai/Mistral-7B-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, model_path)
device = next(model.parameters()).device
print(f"✅ Model loaded on {device}")

# EXPANDED PROMPTS FOR NEUROSCIENCE & PSYCHOTHERAPY
prompt_categories = {
    "🧠 NEUROSCIENCE DEEP DIVE": [
        "Explain neuroplasticity and its implications for therapy.",
        "How do neurotransmitters like dopamine and serotonin influence mental health?",
        "Describe the HPA axis and its role in stress response.",
        "What's the difference between the sympathetic and parasympathetic nervous systems?",
        "How does trauma physically change the brain?",
        "Explain the gut-brain axis and mental health connection.",
        "What are mirror neurons and their role in empathy?",
        "Describe the default mode network's role in depression.",
    ],
    "🛋️ PSYCHOTHERAPY TECHNIQUES": [
        "Compare CBT, DBT, and psychodynamic approaches.",
        "How does EMDR therapy work neurologically?",
        "Explain attachment theory in therapy practice.",
        "What are somatic experiencing techniques?",
        "How does mindfulness change brain function?",
        "Describe internal family systems (IFS) therapy.",
        "What is narrative therapy and how does it work?",
        "Explain acceptance and commitment therapy (ACT).",
    ],
    "⚡ MENTAL HEALTH CONDITIONS": [
        "What's the neurobiological basis of anxiety disorders?",
        "Explain the difference between bipolar I and II disorders.",
        "How does PTSD manifest in brain structure and function?",
        "What are the latest findings on autism spectrum neuroscience?",
        "Describe the neuroinflammation theory of depression.",
        "How does ADHD affect executive functioning in the brain?",
        "What's the relationship between OCD and basal ganglia function?",
        "Explain dissociative disorders from a neural perspective.",
    ],
    "💊 TREATMENT MODALITIES": [
        "How do SSRIs work at the synaptic level?",
        "Compare pharmacological vs. psychotherapeutic approaches for depression.",
        "What are psychedelics' mechanisms of action in therapy?",
        "Explain TMS (transcranial magnetic stimulation) therapy.",
        "How does ketamine work for treatment-resistant depression?",
        "What's the role of neurofeedback in treatment?",
        "Compare MAOIs, SNRIs, and atypical antidepressants.",
        "How do mood stabilizers like lithium work?",
    ],
    "🌍 INTEGRATIVE APPROACHES": [
        "How does nutrition affect mental health from a neuroscientific perspective?",
        "Explain the role of exercise in neurogenesis and mood regulation.",
        "What's the science behind sleep and mental health?",
        "How do social connections impact brain health?",
        "Describe the neuroscience of meditation and its therapeutic benefits.",
        "What role does epigenetics play in mental health?",
        "How does childhood environment shape adult brain function?",
        "Explain the biopsychosocial model of mental health.",
    ],
    "🔬 CUTTING-EDGE RESEARCH": [
        "What are the most promising new directions in neuroscience research?",
        "Explain optogenetics and its potential for mental health treatment.",
        "What's the current understanding of the connectome?",
        "How are AI and machine learning being used in neuroscience?",
        "Describe recent advances in understanding consciousness.",
        "What are brain organoids and how are they used in research?",
        "Explain the potential of CRISPR in neuroscience.",
        "What's the latest on neuroimmunology and mental health?",
    ],
    "🎯 CLINICAL APPLICATIONS": [
        "How do you explain brain-based changes to therapy clients?",
        "What are effective interventions for emotional regulation?",
        "How does therapy actually rewire the brain?",
        "What's the role of the therapeutic alliance in neural change?",
        "How to work with clients who have complex trauma?",
        "What are somatic approaches to anxiety treatment?",
        "How to integrate neuroscience findings into clinical practice?",
        "What are polyvagal theory applications in therapy?",
    ]
}

def generate_response(prompt, max_tokens=300, temperature=0.7):
    """Generate response for a single prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(prompt):
        return response[len(prompt):].strip()
    return response

# Test a selection of prompts from each category
print("🧠 Testing Your Neuroscience & Psychotherapy Expert Model\n")
print("=" * 80)

# Let user choose which categories to test
categories_to_test = list(prompt_categories.keys())
print("Available categories:")
for i, category in enumerate(categories_to_test, 1):
    print(f"{i}. {category}")

choice = input("\nEnter category number to test (or press Enter for all): ")

if choice.strip():
    try:
        selected_idx = int(choice) - 1
        selected_category = categories_to_test[selected_idx]
        categories = {selected_category: prompt_categories[selected_category]}
    except:
        print("Invalid choice, testing all categories.")
        categories = prompt_categories
else:
    categories = prompt_categories

# Generate responses
for category, prompts in categories.items():
    print(f"\n\n{'='*60}")
    print(f"🧠 {category}")
    print(f"{'='*60}")
    
    for i, prompt in enumerate(prompts[:3]):  # Test first 3 from each category
        print(f"\n{i+1}. {prompt}")
        print("-" * 80)
        
        response = generate_response(prompt)
        print(f"💡 {response}")
        print("-" * 80)

# Special bonus: Interactive mode
print("\n\n🎯 INTERACTIVE MODE - Ask your model anything!")
print("Type 'quit' to exit, 'category' to see categories again")

while True:
    user_input = input("\n🧠 Your question: ").strip()
    
    if user_input.lower() == 'quit':
        break
    elif user_input.lower() == 'category':
        for i, category in enumerate(categories_to_test, 1):
            print(f"{i}. {category}")
        continue
    elif not user_input:
        continue
        
    response = generate_response(user_input)
    print(f"\n💡 Model: {response}")