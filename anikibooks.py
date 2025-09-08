import os
import json
import fitz  # PyMuPDF
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

books_folder = r"C:\Users\vladc\OneDrive\Desktop\Books"  # your books folder
output_file = "cleaned_books_dataset.jsonl"

data = []


def extract_epub_text(file_path):
    """Extract text from an EPUB file."""
    try:
        book = epub.read_epub(file_path)
        text_parts = []

        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                try:
                    # Prefer lxml if available
                    try:
                        soup = BeautifulSoup(item.get_content(), "lxml-xml")
                    except Exception:
                        soup = BeautifulSoup(item.get_content(), "html.parser")

                    # Remove unwanted tags
                    for tag in soup(["script", "style", "meta", "link", "nav"]):
                        tag.decompose()

                    text = soup.get_text()
                    text = " ".join(text.split())  # cleanup whitespace
                    if text and len(text) > 100:  # only add substantial text
                        text_parts.append(text)
                except Exception as e:
                    print(f"⚠️ Skipped a section in {file_path}: {e}")
                    continue

        return "\n\n".join(text_parts).strip()
    except Exception as e:
        print(f"❌ Failed to process EPUB {file_path}: {e}")
        return ""


# Check folder validity
if not os.path.exists(books_folder):
    print("Error: The specified path does not exist: " + books_folder)
    exit()
if not os.path.isdir(books_folder):
    print("Error: The specified path is not a directory: " + books_folder)
    exit()

# Process files
pdf_count = 0
epub_count = 0

for root, dirs, files in os.walk(books_folder):
    for filename in files:
        file_path = os.path.join(root, filename)

        if filename.lower().endswith(".pdf"):
            try:
                doc = fitz.open(file_path)
                text = "".join(page.get_text() for page in doc)
                if text.strip():
                    data.append({"text": text.strip(), "file": filename})
                    pdf_count += 1
                    print("✅ Processed PDF: " + filename)
                else:
                    print("⚠️ Empty PDF skipped: " + filename)
            except Exception as e:
                print(f"❌ Failed to process PDF {filename}: {e}")

        elif filename.lower().endswith(".epub"):
            text = extract_epub_text(file_path)
            if text:
                data.append({"text": text, "file": filename})
                epub_count += 1
                print("✅ Processed EPUB: " + filename)
            else:
                print("⚠️ No text extracted from EPUB: " + filename)

# Write output
if data:
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print("\n✅ Processing complete!")
        print("PDF files processed: {}".format(pdf_count))
        print("EPUB files processed: {}".format(epub_count))
        print("Total files: {}".format(len(data)))
        print("Output saved to: {}".format(output_file))
    except Exception as e:
        print("❌ Failed to write output file: " + str(e))
else:
    print("❌ No files were processed. Check if your directory contains PDF or EPUB files.")




