# LLM Paddle OCR Pipeline

A comprehensive OCR pipeline that converts PDF documents to text using PaddleOCR and OpenAI-compatible APIs as fallback.

## Pipeline Overview

The OCR process follows these steps:

1. **PDF to Images** (`pdf_to_img.py`) - Convert PDF pages to individual images
2. **OCR Processing** (`ocr_processor.py`) - Extract text using PaddleOCR 
3. **API Fallback** (`ocr_img_api_fallback.py`) - Use OpenAI-compatible API for difficult images
4. **Text Generation** (`json_to_txt.py`) - Combine JSON results into final text file

## Quick Start

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline
./run_pipeline.sh input.pdf output.txt
```

## Manual Usage

### 1. Convert PDF to Images
```bash
python3 scripts/pdf_to_img.py pdfs/document.pdf images/
```

### 2. Run OCR Processing  
```bash
python3 scripts/ocr_processor.py images/ json_output/
```

### 3. Apply API Fallback (Optional)
```bash
python3 scripts/ocr_img_api_fallback.py json_output/ images/ --api-key YOUR_API_KEY
```

### 4. Generate Final Text
```bash
python3 scripts/json_to_txt.py json_output/ final_output.txt
```

## Configuration

- **Language**: Change OCR language in `scripts/ocr_processor.py` (default: 'ch' for Chinese)
- **API Endpoint**: Configure in `scripts/ocr_img_api_fallback.py` for fallback processing
- **Text Ordering**: Adjust box ordering in `scripts/json_to_txt.py` (top-bottom, left-right, etc.)

## File Prioritization

The pipeline automatically prioritizes `_api.json` files over regular `.json` files for the same page, ensuring the best OCR results are used in the final output.

## Requirements

- Python 3.7+
- PaddleOCR and PaddlePaddle
- OpenAI-compatible API key (for fallback processing)
- PDF documents to process

## Output

The pipeline generates:
- Individual page images (`.png` files)
- OCR results in JSON format
- Final combined text file

## Directory Structure

```
llm-paddle-ocr/
├── scripts/                # Python scripts
│   ├── pdf_to_img.py           # PDF to image conversion
│   ├── ocr_processor.py        # PaddleOCR processing  
│   ├── ocr_img_api_fallback.py # API fallback processing
│   ├── ocr_img_api.py          # API OCR utilities
│   └── json_to_txt.py          # JSON to text conversion
├── run_pipeline.sh         # Complete pipeline script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── pdfs/                   # Input PDF files
├── images/                 # Generated images
├── json_output/            # OCR JSON results
└── texts/                  # Final text outputs
```