#!/usr/bin/env python3
"""
OCR Image Processing via API - Extract text from images using OpenAI API through Poe
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

try:
    import openai
except ImportError:
    print("Error: openai package not found. Install it with: pip install openai")
    sys.exit(1)


def _is_thinking_content(line: str) -> bool:
    """
    Check if a line contains thinking content that should be filtered out
    
    Args:
        line: Text line to check
        
    Returns:
        True if the line is thinking content, False otherwise
    """
    line = line.strip()
    
    # Empty lines
    if not line:
        return True
    
    # Common thinking patterns
    thinking_patterns = [
        "*Thinking*",
        "*thinking*", 
        "*Thinking...*",
        "*thinking...*",
        "**Thinking**",
        "**thinking**",
    ]
    
    # Check for exact thinking markers
    for pattern in thinking_patterns:
        if line == pattern:
            return True
    
    # Lines that start with > (thinking explanations)
    if line.startswith(">"):
        return True
    
    # Lines that start with thinking keywords followed by explanations
    thinking_keywords = [
        "I'm now", "I am now", "I'm", "I am",
        "Let me", "I need to", "I should", "I will",
        "My current focus", "I've focused", "I have focused",
        "Processing", "Analyzing", "Examining", "Looking at",
        "Refining", "Decoding", "Dissecting", "Comparing"
    ]
    
    for keyword in thinking_keywords:
        if line.startswith(keyword):
            return True
    
    # Lines with thinking section headers (often in markdown format)
    if line.startswith("**") and line.endswith("**") and len(line) > 4:
        # Check if it's a thinking section header
        inner_text = line[2:-2].strip()
        thinking_headers = [
            "Processing", "Analyzing", "Examining", "Thinking", 
            "Refining", "Decoding", "Understanding", "Comparing",
            "Text Extraction", "Character Arrangement", "Image Content"
        ]
        for header in thinking_headers:
            if header.lower() in inner_text.lower():
                return True
    
    return False


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ocr_image_via_api(image_path: str, api_key: str, base_url: str = "https://api.poe.com/v1", 
                     model: str = "GPT-OSS-20B-T", prompt: str = "請識別圖片中的文字。請不要提供任何別的內容或解釋，只返回文字。",
                     filter_thinking: bool = True) -> Dict[str, Any]:
    """
    Extract text from an image using OpenAI-style API and return in PaddleOCR format
    
    Args:
        image_path: Path to the image file
        api_key: API key for the service
        base_url: Base URL for the API endpoint (default: Poe endpoint)
        model: Model to use for OCR (default: GPT-OSS-20B-T)
        prompt: Prompt to send with the image (default: Chinese OCR prompt)
        filter_thinking: Whether to filter out thinking content from thinking models (default: True)
        
    Returns:
        Dictionary in PaddleOCR format with regions (without bbox coordinates)
    """
    # Validate image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Encode image to base64
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        raise ValueError(f"Failed to encode image: {e}")
    
    # Initialize OpenAI client with configurable endpoint
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Prepare the message with image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    try:
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        
        # Extract text response
        extracted_text = response.choices[0].message.content
        if not extracted_text:
            extracted_text = ""
        
        # Convert to PaddleOCR-compatible JSON format
        # Split text into lines and create regions without bbox coordinates
        lines = extracted_text.strip().split('\n')
        regions = []
        
        # Filter out thinking content from thinking models (if enabled)
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if line and (not filter_thinking or not _is_thinking_content(line)):
                filtered_lines.append(line)
        
        for i, line in enumerate(filtered_lines):
            regions.append({
                "region_id": i,
                "text": line,
                "bbox": [],  # Empty array for fallback mode
                "confidence": None  # No confidence available from API
            })
        
        # Create the result dictionary in PaddleOCR format
        result = {
            "input_path": str(image_path),
            "timestamp": time.time(),
            "total_regions": len(regions),
            "regions": regions,
            "api_fallback": True,  # Flag to indicate this was processed via API fallback
            "api_model": model,
            "api_base_url": base_url,
            "thinking_filter_enabled": filter_thinking,
            "original_response_lines": len(lines),  # Total lines before filtering
            "filtered_response_lines": len(filtered_lines)  # Lines after filtering
        }
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")


def main():
    """Command line interface for OCR via API"""
    parser = argparse.ArgumentParser(description="Extract text from images using OpenAI-style API")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--api-key", required=True, help="API key for the service")
    parser.add_argument("--base-url", default="https://api.poe.com/v1", 
                        help="Base URL for the API endpoint (default: Poe endpoint)")
    parser.add_argument("--model", default="GPT-OSS-20B-T", 
                        help="Model to use for OCR (default: GPT-OSS-20B-T)")
    parser.add_argument("--prompt", default="請識別圖片中的文字。請不要提供任何別的內容或解釋，只返回文字。",
                        help="Prompt to send with the image")
    parser.add_argument("--no-filter-thinking", action="store_true",
                        help="Disable filtering of thinking content from thinking models")
    parser.add_argument("--output", "-o", help="Output file to save results (.json or .txt)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    if args.verbose:
        print(f"Processing image: {image_path}")
        print(f"Using model: {args.model}")
        print(f"Base URL: {args.base_url}")
    
    try:
        # Extract text from image
        filter_thinking = not args.no_filter_thinking
        result = ocr_image_via_api(str(image_path), args.api_key, args.base_url, args.model, args.prompt, filter_thinking)
        
        if args.output:
            # Save to file
            output_path = Path(args.output)
            
            if output_path.suffix.lower() == '.json':
                # Save as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"✅ OCR complete. JSON saved to {output_path}")
            else:
                # Save as text (extract text from regions)
                text_lines = [region["text"] for region in result["regions"]]
                extracted_text = '\n'.join(text_lines)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                print(f"✅ OCR complete. Text saved to {output_path}")
                
            if args.verbose:
                print(f"Total regions: {result['total_regions']}")
        else:
            # Print to stdout
            print("=" * 50)
            print("EXTRACTED TEXT:")
            print("=" * 50)
            text_lines = [region["text"] for region in result["regions"]]
            print('\n'.join(text_lines))
            print("=" * 50)
            print(f"Total regions: {result['total_regions']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


class ApiFallbackFunction:
    """
    Picklable API fallback function class for use with multiprocessing
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.poe.com/v1", 
                 model: str = "GPT-OSS-20B-T", 
                 prompt: str = "請識別圖片中的文字。請不要提供任何別的內容或解釋，只返回文字。",
                 filter_thinking: bool = True):
        """
        Initialize API fallback function with configuration
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API endpoint (default: Poe endpoint)
            model: Model to use for OCR (default: GPT-OSS-20B-T)
            prompt: Prompt to send with the image (default: Chinese OCR prompt)
            filter_thinking: Whether to filter out thinking content from thinking models (default: True)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt = prompt
        self.filter_thinking = filter_thinking
    
    def __call__(self, image_path: str) -> Dict[str, Any]:
        """
        Fallback function that calls the API OCR with configured parameters
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary in PaddleOCR format with regions
        """
        return ocr_image_via_api(image_path, self.api_key, self.base_url, self.model, self.prompt, self.filter_thinking)


def create_api_fallback_function(api_key: str, base_url: str = "https://api.poe.com/v1", 
                                model: str = "GPT-OSS-20B-T", 
                                prompt: str = "請識別圖片中的文字。請不要提供任何別的內容或解釋，只返回文字。",
                                filter_thinking: bool = True):
    """
    Create a fallback function configured with API parameters for use with OCRProcessor
    
    Args:
        api_key: API key for the service
        base_url: Base URL for the API endpoint (default: Poe endpoint)
        model: Model to use for OCR (default: GPT-OSS-20B-T)
        prompt: Prompt to send with the image (default: Chinese OCR prompt)
        filter_thinking: Whether to filter out thinking content from thinking models (default: True)
        
    Returns:
        Callable that takes image_path and returns OCR result dict
    """
    return ApiFallbackFunction(api_key, base_url, model, prompt, filter_thinking)


if __name__ == "__main__":
    exit(main())
