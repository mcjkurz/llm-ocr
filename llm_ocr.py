#!/usr/bin/env python3
"""
LLM OCR Tool - Extract text from PDF documents using LLM API endpoints
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Generator, Tuple
import base64
import io

import fitz  # PyMuPDF
import requests
from PIL import Image


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMOCRProcessor:
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        """
        Initialize the LLM OCR processor
        
        Args:
            api_endpoint: OpenAI-compatible API endpoint
            api_key: API key for authentication (optional)
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.chinese_prompt = "请识别并提取这个图片中的所有中文文字"
        
    def pdf_pages_generator(self, pdf_path: str):
        """
        Generator that yields PDF pages as images one at a time
        
        Args:
            pdf_path: Path to the PDF file
            
        Yields:
            Tuple of (page_number, PIL.Image, total_pages)
        """
        logger.info(f"Opening PDF for processing: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages")
            
            for page_num in range(total_pages):
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                page = doc.load_page(page_num)
                
                # Convert to image with high DPI for better OCR quality
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                yield (page_num + 1, img, total_pages)
                
                # Clean up page resources immediately
                pix = None
                page = None
                
            doc.close()
            logger.info("PDF processing completed")
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True, quality=95)
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')
    
    def ocr_image(self, image: Image.Image) -> str:
        """
        Extract text from image using LLM API
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        base64_image = self.image_to_base64(image)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": "gpt-4-vision-preview",  # Default model, can be changed
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.chinese_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(
                f"{self.api_endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.error("Unexpected API response format")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error processing API response: {e}")
            return ""
    
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> None:
        """
        Process PDF file and save extracted text to separate files
        Uses generator pattern for memory efficiency
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory (optional, defaults to PDF filename-based dir)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return
        
        # Create output directory
        if output_dir is None:
            output_dir = pdf_path.stem + "_ocr_output"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        # Process each page using generator (memory efficient)
        try:
            for page_num, image, total_pages in self.pdf_pages_generator(str(pdf_path)):
                logger.info(f"Extracting text from page {page_num}/{total_pages}")
                
                # Extract text using LLM
                extracted_text = self.ocr_image(image)
                
                if extracted_text:
                    # Save to text file
                    output_file = output_path / f"page_{page_num:03d}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    logger.info(f"Saved extracted text to: {output_file}")
                else:
                    logger.warning(f"No text extracted from page {page_num}")
                
                # Clean up image from memory immediately
                image.close()
                
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            return
        
        logger.info(f"Processing completed. Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF documents using LLM API endpoints"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--api-endpoint",
        required=True,
        help="OpenAI-compatible API endpoint (e.g., https://api.openai.com/v1)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (can also use OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for extracted text files (default: PDF_filename_ocr_output)"
    )
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        logger.warning("No API key provided. Some APIs may require authentication.")
    
    # Initialize processor
    processor = LLMOCRProcessor(
        api_endpoint=args.api_endpoint,
        api_key=api_key
    )
    
    # Process PDF
    processor.process_pdf(args.pdf_path, args.output_dir)


if __name__ == "__main__":
    main()
