# LLM OCR

A simple tool to extract text from PDF documents using LLM API endpoints. The tool converts PDF pages to images and sends them to an OpenAI-compatible API for OCR processing.

## Features

- Convert PDF documents to individual page images
- Extract text from images using LLM vision capabilities
- Save extracted text as separate `.txt` files
- Support for Chinese text recognition
- Simple command-line interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/mcjkurz/llm-ocr.git
cd llm-ocr
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python llm_ocr.py document.pdf --api-endpoint https://api.openai.com/v1 --api-key your-api-key
```

With environment variable for API key:
```bash
export OPENAI_API_KEY=your-api-key
python llm_ocr.py document.pdf --api-endpoint https://api.openai.com/v1
```

Custom output directory:
```bash
python llm_ocr.py document.pdf --api-endpoint https://api.openai.com/v1 --output-dir my_output
```

## Arguments

- `pdf_path`: Path to the PDF file to process (required)
- `--api-endpoint`: OpenAI-compatible API endpoint (required)
- `--api-key`: API key for authentication (optional if OPENAI_API_KEY env var is set)
- `--output-dir`: Output directory for extracted text files (optional, defaults to `{pdf_filename}_ocr_output`)

## Output

The tool creates a directory containing separate `.txt` files for each page:
```
document_ocr_output/
├── page_001.txt
├── page_002.txt
└── page_003.txt
```

## Requirements

- Python 3.7+
- OpenAI-compatible API endpoint with vision capabilities
- PDF files to process

## License

MIT License
