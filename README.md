# Anonero

Fine-tuning experiments with [Mistral](https://mistral.ai) on book datasets.  
Includes scripts for chunking, preprocessing, and running training with QLoRA.

## Structure
- `chunk_processor.py` — splits large datasets into smaller chunks for training.
- `books_chunked.jsonl` — processed dataset (chunked).
- `cleaned_books_dataset.jsonl` — preprocessed dataset before chunking.
- `mistral-books-finetuned/` — checkpoints and adapters from fine-tuning.

## Notes
- Some files are large (>50MB). Git LFS is recommended if you plan to clone/push.
- Training was done locally with Hugging Face + QLoRA.

## Todo
- Add instructions for running the fine-tuning pipeline.
- Clean up and optimize preprocessing scripts.
- Experiment with different datasets and evaluation.
