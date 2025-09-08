from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import gradio as gr
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load your fine-tuned model (do this once at startup)
model_path = r"C:\Users\vladc\OneDrive\Desktop\anonero.py\mistral-books-finetuned"
base_model_id = "mistralai/Mistral-7B-v0.1"

print("Loading model...")
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
device = next(model.parameters()).device
print(f"✅ Model loaded on {device}")

def generate_response(message, history, max_tokens=300, temperature=0.7):
    """Generate response using your fine-tuned model"""
    
    # Create prompt from conversation history
    prompt = message
    
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
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
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated text (remove the prompt)
    if full_response.startswith(prompt):
        response = full_response[len(prompt):].strip()
    else:
        response = full_response
    
    return response

def chat_interface(message, history, max_tokens, temperature):
    """Gradio chat interface function"""
    try:
        response = generate_response(message, history, max_tokens, temperature)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Neuro-Therapy AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Neuro-Therapy AI Assistant
    *Your fine-tuned neuroscience and psychotherapy expert*
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Ask me about neuroscience, therapy, mental health...",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Conversation")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            max_tokens = gr.Slider(
                minimum=50,
                maximum=500,
                value=250,
                step=50,
                label="Response Length"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Creativity (Temperature)"
            )
            
            gr.Markdown("### 💡 Prompt Examples")
            examples = gr.Examples(
                examples=[
                    "Explain neuroplasticity in simple terms",
                    "What are the key principles of CBT?",
                    "How does trauma affect brain development?",
                    "Describe the neuroscience of mindfulness",
                    "What's the difference between bipolar I and II?"
                ],
                inputs=msg,
                label="Click to try:"
            )
    
    # Event handlers
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history, max_tokens, temperature):
        user_message = history[-1][0]
        bot_message = chat_interface(user_message, history, max_tokens, temperature)
        history[-1][1] = bot_message
        return history
    
    # Connect components
    msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        fn=bot,
        inputs=[chatbot, max_tokens, temperature],
        outputs=chatbot
    )
    
    submit_btn.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        fn=bot,
        inputs=[chatbot, max_tokens, temperature],
        outputs=chatbot
    )
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    
    gr.Markdown("""
    ---
    *Model fine-tuned on neuroscience and psychotherapy materials. Responses may vary based on temperature settings.*
    """)

# Launch the app WITH SHARING ENABLED
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # 🚀 THIS ENABLES GRADIO LIVE SHARING
    )