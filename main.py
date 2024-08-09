import PyPDF2
from transformers import pipeline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text

def split_text(text, max_length=1024):
    sentences = text.split('. ')
    chunks = []
    chunk = ''
    for sentence in sentences:
        if len(chunk) + len(sentence) + 1 <= max_length:
            chunk += sentence + '. '
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '. '
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def summarize_text(text, max_length=300, min_length=80):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary_text = summary[0]['summary_text']
    
    points = summary_text.split('. ')
    formatted_summary = "\n".join(f"- {point.strip()}" for point in points if point)
    
    return formatted_summary

def summarize_large_text(text, chunk_size=1024):
    chunks = split_text(text, max_length=chunk_size)
    summaries = [summarize_text(chunk, max_length=200, min_length=80) for chunk in chunks]
    return "\n\n".join(summaries)

def save_summary_to_file(summary, output_path="summary.txt"):
    with open(output_path, 'w') as file:
        file.write(summary)

pdf_path = 'file.pdf'
extracted_text = extract_text_from_pdf(pdf_path)
summary = summarize_large_text(extracted_text, chunk_size=1024)
# print(summary)
save_summary_to_file(summary, output_path="summary.txt")