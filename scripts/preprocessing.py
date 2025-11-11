import json
def clear_processed_dir():
    for filename in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

import os
import glob
from docx import Document
import pdfplumber
import nltk
from pypdf import PdfReader

# Descargar recursos de NLTK si no están
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
RAW_DIR = 'data/info/raw'
PROCESSED_DIR = 'data/info/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_text_from_word(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_pdf(file_path):
    text = ''
    import re
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            try:
                page_text = page.extract_text(layout=True)
            except TypeError:
                page_text = page.extract_text()
            if page_text:
                page_text = page_text.replace('\r\n', '\n').replace('\r', '\n')
                page_text = re.sub(r'\n{3,}', '\n\n', page_text)
                text += page_text + '\n'
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def get_pdf_bookmarks(file_path):
    reader = PdfReader(file_path)
    bookmarks = reader.outline if hasattr(reader, 'outline') else []
    bookmark_info = []
    from pypdf.generic import Destination
    for item in bookmarks:
        if isinstance(item, Destination):
            title = item.title
            page_number = reader.get_destination_page_number(item)
            bookmark_info.append({"title": title, "start_page": page_number})
        elif isinstance(item, list):
            parent_bookmark = item[0]
            title = parent_bookmark.title
            page_number = reader.get_destination_page_number(parent_bookmark)
            bookmark_info.append({"title": title, "start_page": page_number})
    return bookmark_info, len(reader.pages)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text, chunk_size=500):
    import re
    # Normalizar saltos de línea y espacios
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Detectar títulos en el texto (para TXT/Word)
    def detect_titles(lines):
        titles = []
        for idx, line in enumerate(lines):
            is_title = (
                (len(line) < 50 and not any(c in line for c in '.!?')) or
                (sum(1 for c in line if c.isupper()) > len(line) // 2)
            )
            if is_title:
                titles.append(idx)
        return titles

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    title_indices = detect_titles(lines)
    if title_indices:
        chunks = []
        for i, idx in enumerate(title_indices):
            start = idx
            end = title_indices[i+1] if i+1 < len(title_indices) else len(lines)
            chunk = '\n'.join(lines[start:end]).strip()
            if chunk:
                chunks.append(chunk)
        if chunks:
            return chunks

    # Si no hay títulos, usar el método por párrafos
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    # Si no, usar el método actual por frases
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c.strip()]

def process_files():
    clear_processed_dir()
    patterns = [
        os.path.join(RAW_DIR, '*.word'),
        os.path.join(RAW_DIR, '*.docx'),
        os.path.join(RAW_DIR, '*.txt'),
        os.path.join(RAW_DIR, '*.pdf'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        summary = []
        if ext == '.pdf':
            # Intentar chunking por bookmarks
            bookmark_info, num_pages = get_pdf_bookmarks(file_path)
            if bookmark_info:
                reader = PdfReader(file_path)
                for i, item in enumerate(bookmark_info):
                    start_page = item['start_page']
                    end_page = bookmark_info[i+1]['start_page']-1 if i+1 < len(bookmark_info) else num_pages-1
                    text = ''
                    with pdfplumber.open(file_path) as pdf:
                        for page_num in range(start_page, end_page+1):
                            page = pdf.pages[page_num]
                            page_text = page.extract_text(layout=True) if hasattr(page, 'extract_text') else ''
                            if page_text:
                                text += page_text + '\n'
                    chunk_filename = f'{base_name}_chunk_{i+1}_{item["title"]}.txt'.replace(' ', '_')
                    out_file = os.path.join(PROCESSED_DIR, chunk_filename)
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(text.strip())
                    summary.append({
                        "chunk_file": chunk_filename,
                        "title": item["title"],
                        "start_page": start_page,
                        "end_page": end_page
                    })
                continue
            else:
                text = extract_text_from_pdf(file_path)
        elif ext in ['.word', '.docx']:
            text = extract_text_from_word(file_path)
        elif ext == '.txt':
            text = extract_text_from_txt(file_path)
        else:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_filename = f'{base_name}_chunk_{i+1}.txt'
            out_file = os.path.join(PROCESSED_DIR, chunk_filename)
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            summary.append({
                "chunk_file": chunk_filename,
                "title": chunk.split('\n')[0][:50],
                "start": i+1
            })

if __name__ == '__main__':
    process_files()