#!/usr/bin/env python3
"""
LLM OCR Tool - Extract text from PDF documents using LLM API endpoints
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
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
        
    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL Images
        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert to image with high DPI for better OCR quality
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
                logger.info(f"Converted page {page_num + 1}/{len(doc)}")
            doc.close()
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
            
        return images
    
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
        
        # Convert PDF to images
        try:
            images = self.pdf_to_images(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return
        
        # Process each page
        for i, image in enumerate(images):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{len(images)}")
            
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
