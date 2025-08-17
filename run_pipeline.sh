#!/bin/bash

# LLM OCR Pipeline Runner
# Usage: ./run_pipeline.sh input.pdf output.txt [api_key]

set -e  # Exit on any error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_pdf> <output_txt> [api_key]"
    echo "Example: $0 pdfs/document.pdf output.txt sk-your-api-key"
    exit 1
fi

INPUT_PDF="$1"
OUTPUT_TXT="$2"
API_KEY="${3:-}"

# Check if input PDF exists
if [ ! -f "$INPUT_PDF" ]; then
    echo "Error: Input PDF file '$INPUT_PDF' not found"
    exit 1
fi

# Get base name for intermediate directories
BASENAME=$(basename "$INPUT_PDF" .pdf)
IMAGES_DIR="images/$BASENAME"
JSON_DIR="json_output/$BASENAME"

echo "🚀 Starting OCR Pipeline for: $INPUT_PDF"
echo "📁 Images will be saved to: $IMAGES_DIR"
echo "📄 JSON results will be saved to: $JSON_DIR"
echo "📝 Final text will be saved to: $OUTPUT_TXT"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Warning: Virtual environment not found. Make sure dependencies are installed."
fi

# Create output directories
mkdir -p "$IMAGES_DIR"
mkdir -p "$JSON_DIR"
mkdir -p "$(dirname "$OUTPUT_TXT")"

echo ""
echo "Step 1/4: Converting PDF to images..."
python3 scripts/pdf_to_img.py "$INPUT_PDF" "$IMAGES_DIR"

echo ""
echo "Step 2/4: Running OCR with PaddleOCR..."
python3 scripts/ocr_processor.py "$IMAGES_DIR" "$JSON_DIR"

echo ""
echo "Step 3/4: Applying API fallback (if needed)..."
if [ -n "$API_KEY" ]; then
    echo "🔑 Using provided API key for fallback processing"
    python3 scripts/ocr_img_api_fallback.py "$JSON_DIR" "$IMAGES_DIR" --api-key "$API_KEY"
elif [ -n "$OPENAI_API_KEY" ]; then
    echo "🔑 Using OPENAI_API_KEY environment variable"
    python3 scripts/ocr_img_api_fallback.py "$JSON_DIR" "$IMAGES_DIR" --api-key "$OPENAI_API_KEY"
else
    echo "⚠️  No API key provided. Skipping API fallback processing."
    echo "   (Set OPENAI_API_KEY environment variable or pass as 3rd argument)"
fi

echo ""
echo "Step 4/4: Generating final text file..."
python3 scripts/json_to_txt.py "$JSON_DIR" "$OUTPUT_TXT"

echo ""
echo "✅ Pipeline completed successfully!"
echo "📝 Final text saved to: $OUTPUT_TXT"

# Show file size and line count
if [ -f "$OUTPUT_TXT" ]; then
    FILE_SIZE=$(wc -c < "$OUTPUT_TXT")
    LINE_COUNT=$(wc -l < "$OUTPUT_TXT")
    echo "📊 Output: $LINE_COUNT lines, $FILE_SIZE bytes"
fi

echo ""
echo "🗂️  Intermediate files:"
echo "   Images: $IMAGES_DIR"
echo "   JSON:   $JSON_DIR"
echo ""
echo "To clean up intermediate files, run:"
echo "   rm -rf $IMAGES_DIR $JSON_DIR"
