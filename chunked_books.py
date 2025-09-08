import json
from transformers import AutoTokenizer

# pick your model tokenizer
model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def chunk_text(text, max_tokens=1024, overlap=100):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)
        i += max_tokens - overlap

    return chunks

input_file = "cleaned_books_dataset.jsonl"
output_file = "books_chunked.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        entry = json.loads(line)
        text = entry["text"]
        file_name = entry.get("file", "unknown")

        for chunk in chunk_text(text, max_tokens=1024, overlap=100):
            if chunk.strip():
                f_out.write(json.dumps({"text": chunk, "file": file_name}, ensure_ascii=False) + "\n")

print(f"✅ Done! Chunked dataset saved to {output_file}")
