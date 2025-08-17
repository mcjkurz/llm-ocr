#!/usr/bin/env python3
"""
OCR Image API Fallback and Confidence Filtering

This module processes JSON files from PaddleOCR and either:
1. If the file needs API fallback (based on difficulty assessment), calls OpenAI-compatible API and creates _api.json
2. If no fallback needed, applies confidence thresholding to filter low-confidence regions

Includes integrated API OCR functionality using OpenAI-compatible endpoints (like Poe).
"""

import os
import json
import argparse
import logging
import time
import base64
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Import OpenAI for API calls
try:
    import openai
    API_FALLBACK_AVAILABLE = True
except ImportError:
    API_FALLBACK_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    Create a fallback function configured with API parameters
    
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


def assess_image_difficulty(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess OCR difficulty and determine if fallback is needed
    
    Args:
        ocr_result: OCR result dictionary with 'total_regions' and 'regions'
    
    Returns:
        Dictionary with difficulty assessment:
        {
            'should_fallback': bool,
            'reasons': List[str],
            'total_regions': int,
            'low_confidence_regions': int,
            'short_text_regions': int
        }
    """
    total_regions = ocr_result.get('total_regions', 0)
    regions = ocr_result.get('regions', [])
    
    reasons = []
    
    # Heuristic 1: More than 20 regions
    if total_regions > 20:
        reasons.append(f"Too many regions: {total_regions} > 20")
    
    # Heuristic 2: Five or more regions with 2 or less characters
    short_text_regions = sum(1 for region in regions if len(region.get('text', '')) <= 2)
    if short_text_regions >= 5:
        reasons.append(f"Too many short text regions: {short_text_regions} >= 5")
    
    # Heuristic 3: 25% or more regions have confidence below 0.5
    low_confidence_regions = sum(1 for region in regions 
                                if region.get('confidence', 1.0) is not None 
                                and region.get('confidence', 1.0) < 0.5)
    
    if total_regions > 0:
        low_confidence_ratio = low_confidence_regions / total_regions
        if low_confidence_ratio >= 0.25:
            reasons.append(f"Too many low confidence regions: {low_confidence_ratio:.1%} >= 25%")
    
    should_fallback = len(reasons) > 0
    
    return {
        'should_fallback': should_fallback,
        'reasons': reasons,
        'total_regions': total_regions,
        'low_confidence_regions': low_confidence_regions,
        'short_text_regions': short_text_regions
    }


def filter_low_confidence_regions(ocr_result: Dict[str, Any], 
                                confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Filter out regions with confidence below threshold
    
    Args:
        ocr_result: OCR result dictionary with 'regions'
        confidence_threshold: Minimum confidence threshold (default: 0.5)
    
    Returns:
        Filtered OCR result dictionary
    """
    regions = ocr_result.get('regions', [])
    
    # Filter regions with confidence >= threshold
    filtered_regions = []
    for region in regions:
        confidence = region.get('confidence')
        if confidence is None or confidence >= confidence_threshold:
            filtered_regions.append(region)
    
    # Update result
    filtered_result = ocr_result.copy()
    filtered_result['regions'] = filtered_regions
    filtered_result['total_regions'] = len(filtered_regions)
    filtered_result['confidence_filtering_applied'] = True
    filtered_result['confidence_threshold'] = confidence_threshold
    filtered_result['original_total_regions'] = len(regions)
    filtered_result['filtered_out_regions'] = len(regions) - len(filtered_regions)
    
    return filtered_result


def find_matching_image(json_path: Path, image_dir: Path, 
                       image_extensions: List[str] = None) -> Optional[Path]:
    """
    Find the corresponding image file for a JSON file
    
    Args:
        json_path: Path to the JSON file
        image_dir: Directory containing image files
        image_extensions: List of image extensions to search for
    
    Returns:
        Path to the matching image file, or None if not found
    """
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg']
    
    json_stem = json_path.stem
    
    # Try each extension
    for ext in image_extensions:
        image_path = image_dir / f"{json_stem}{ext}"
        if image_path.exists():
            return image_path
        # Also try uppercase extensions
        image_path = image_dir / f"{json_stem}{ext.upper()}"
        if image_path.exists():
            return image_path
    
    return None


def create_api_result_for_image(image_path: Path, original_json_data: Dict[str, Any], 
                               api_fallback_function, difficulty_assessment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create API OCR result for an image using fallback function
    
    Args:
        image_path: Path to the image file
        original_json_data: Original PaddleOCR JSON data
        api_fallback_function: Function to call for API OCR
        difficulty_assessment: Assessment results from original OCR
    
    Returns:
        API OCR result dictionary with metadata
    """
    try:
        # Call API fallback function
        api_result = api_fallback_function(str(image_path))
        
        # Add metadata about why API was used and original stats
        api_result["api_fallback_used"] = True
        api_result["api_fallback_reasons"] = difficulty_assessment['reasons']
        api_result["original_paddleocr_stats"] = {
            "total_regions": difficulty_assessment['total_regions'],
            "low_confidence_regions": difficulty_assessment['low_confidence_regions'],
            "short_text_regions": difficulty_assessment['short_text_regions']
        }
        api_result["original_paddleocr_timestamp"] = original_json_data.get("timestamp")
        
        return api_result
        
    except Exception as e:
        # If API fails, create error result
        error_result = {
            "input_path": str(image_path),
            "timestamp": time.time(),
            "total_regions": 0,
            "regions": [],
            "api_fallback_used": True,
            "api_fallback_failed": True,
            "api_fallback_error": str(e),
            "api_fallback_reasons": difficulty_assessment['reasons'],
            "original_paddleocr_stats": {
                "total_regions": difficulty_assessment['total_regions'],
                "low_confidence_regions": difficulty_assessment['low_confidence_regions'],
                "short_text_regions": difficulty_assessment['short_text_regions']
            }
        }
        return error_result


def process_json_files_for_api_fallback(
    json_dir: str,
    image_dir: str,
    output_dir: str,
    api_fallback_function,
    confidence_threshold: float = 0.5,
    image_extensions: List[str] = None,
    overwrite_existing: bool = False
) -> Dict[str, Any]:
    """
    Process JSON files: apply API fallback if needed, otherwise apply confidence filtering
    
    Args:
        json_dir: Directory containing JSON files from PaddleOCR
        image_dir: Directory containing original image files  
        output_dir: Directory to save processed JSON outputs (can be same as json_dir)
        api_fallback_function: Function to call for API OCR (can be None if no API processing desired)
        confidence_threshold: Confidence threshold for filtering (default: 0.5)
        image_extensions: List of image extensions to search for
        overwrite_existing: Whether to overwrite existing processed files
    
    Returns:
        Dictionary containing processing summary
    """
    json_dir = Path(json_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg']
    
    # Find all JSON files (excluding _api.json files)
    json_files = [f for f in json_dir.glob("*.json") if not f.name.endswith("_api.json")]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    results = {
        "total_files": len(json_files),
        "needs_api_fallback": 0,
        "api_success": 0,
        "api_failed": 0,
        "confidence_filtered": 0,
        "skipped_no_image": 0,
        "skipped_existing": 0,
        "processed_files": [],
        "failed_files": [],
        "no_image_files": []
    }
    
    # Phase 1: Analyze all files and categorize them
    logger.info("Analyzing files to determine processing requirements...")
    files_for_api = []
    files_for_confidence_filtering = []
    
    with tqdm(total=len(json_files), desc="Analyzing JSON files", unit="file") as pbar:
        for json_file in json_files:
            try:
                # Load original JSON data
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Assess difficulty
                difficulty_assessment = assess_image_difficulty(json_data)
                
                if not difficulty_assessment['should_fallback']:
                    # Add to confidence filtering list
                    files_for_confidence_filtering.append((json_file, json_data, difficulty_assessment))
                else:
                    # Need API fallback - check if we should process it
                    results["needs_api_fallback"] += 1
                    
                    # Skip if no API function provided
                    if api_fallback_function is None:
                        pbar.update(1)
                        continue
                    
                    # Check if API output already exists
                    api_json_name = f"{json_file.stem}_api.json"
                    api_json_path = output_dir / api_json_name
                    
                    if api_json_path.exists() and not overwrite_existing:
                        results["skipped_existing"] += 1
                        pbar.update(1)
                        continue
                    
                    # Find corresponding image
                    image_path = find_matching_image(json_file, image_dir, image_extensions)
                    if image_path is None:
                        results["skipped_no_image"] += 1
                        results["no_image_files"].append(str(json_file))
                        logger.warning(f"No matching image found for {json_file}")
                        pbar.update(1)
                        continue
                    
                    # Add to API processing list
                    files_for_api.append((json_file, json_data, difficulty_assessment, image_path, api_json_path))
                
            except Exception as e:
                results["api_failed"] += 1
                results["failed_files"].append({
                    "json_file": str(json_file),
                    "error": str(e)
                })
                logger.error(f"Error analyzing {json_file}: {e}")
            
            pbar.update(1)
    
    # Phase 2: Apply confidence filtering to files that don't need API
    if files_for_confidence_filtering:
        logger.info(f"Applying confidence filtering to {len(files_for_confidence_filtering)} files...")
        with tqdm(total=len(files_for_confidence_filtering), desc="Applying confidence filtering", unit="file") as pbar:
            for json_file, json_data, difficulty_assessment in files_for_confidence_filtering:
                try:
                    # Apply confidence filtering
                    filtered_data = filter_low_confidence_regions(json_data, confidence_threshold)
                    
                    # Save filtered version (overwrite original or save to output dir)
                    if output_dir == json_dir:
                        # Overwrite original file
                        output_path = json_file
                    else:
                        # Save to output directory
                        output_path = output_dir / json_file.name
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                    
                    results["confidence_filtered"] += 1
                    results["processed_files"].append({
                        "json_file": str(json_file),
                        "output_file": str(output_path),
                        "processing_type": "confidence_filtering",
                        "original_regions": filtered_data.get("original_total_regions", 0),
                        "filtered_regions": filtered_data.get("total_regions", 0),
                        "removed_regions": filtered_data.get("filtered_out_regions", 0)
                    })
                    
                except Exception as e:
                    results["api_failed"] += 1
                    results["failed_files"].append({
                        "json_file": str(json_file),
                        "error": str(e)
                    })
                    logger.error(f"Error applying confidence filtering to {json_file}: {e}")
                
                pbar.update(1)
    
    # Phase 3: Process files that need API calls
    if files_for_api:
        logger.info(f"Processing {len(files_for_api)} files with API calls...")
        with tqdm(total=len(files_for_api), desc="Processing API calls", unit="file") as pbar:
            for json_file, json_data, difficulty_assessment, image_path, api_json_path in files_for_api:
                try:
                    # Create API result
                    api_result = create_api_result_for_image(
                        image_path, json_data, api_fallback_function, difficulty_assessment
                    )
                    
                    # Save API result
                    with open(api_json_path, 'w', encoding='utf-8') as f:
                        json.dump(api_result, f, ensure_ascii=False, indent=2)
                    
                    # Track results
                    if api_result.get("api_fallback_failed", False):
                        results["api_failed"] += 1
                        results["failed_files"].append({
                            "json_file": str(json_file),
                            "image_file": str(image_path),
                            "api_output": str(api_json_path),
                            "error": api_result.get("api_fallback_error", "Unknown error")
                        })
                    else:
                        results["api_success"] += 1
                        results["processed_files"].append({
                            "json_file": str(json_file),
                            "image_file": str(image_path),
                            "output_file": str(api_json_path),
                            "processing_type": "api_fallback",
                            "reasons": difficulty_assessment['reasons'],
                            "api_regions": api_result.get("total_regions", 0)
                        })
                    
                except Exception as e:
                    results["api_failed"] += 1
                    results["failed_files"].append({
                        "json_file": str(json_file),
                        "image_file": str(image_path),
                        "error": str(e)
                    })
                    logger.error(f"Error processing API call for {json_file}: {e}")
                
                pbar.update(1)
    
    return results


def main():
    """Command line interface for API fallback and confidence filtering"""
    parser = argparse.ArgumentParser(description="Process PaddleOCR results: apply API fallback or confidence filtering")
    parser.add_argument("json_dir", help="Directory containing JSON files from PaddleOCR")
    parser.add_argument("image_dir", help="Directory containing original image files")
    parser.add_argument("--output-dir", help="Output directory for processed JSON files (default: same as json_dir)")
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"],
                        help="Image file extensions to search for (default: .png .jpg .jpeg)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing processed files")
    
    # Confidence filtering
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Confidence threshold for filtering regions (default: 0.5)")
    
    # API configuration arguments (optional)
    parser.add_argument("--api-key",
                        help="API key for fallback OCR service (optional - without this, only confidence filtering is applied)")
    parser.add_argument("--api-base-url", default="https://api.poe.com/v1",
                        help="Base URL for API fallback (default: Poe endpoint)")
    parser.add_argument("--api-model", default="GPT-OSS-20B-T",
                        help="Model for API fallback (default: GPT-OSS-20B-T)")
    parser.add_argument("--api-prompt", default="請識別圖片中的文字。請不要提供任何別的內容或解釋，只返回文字。",
                        help="Prompt for API fallback")
    parser.add_argument("--no-filter-thinking", action="store_true",
                        help="Disable filtering of thinking content from thinking models")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Suppress HTTP request logging from OpenAI/httpx client and related libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.json_dir
    
    try:
        # Create API fallback function if API key is provided
        api_fallback_function = None
        if args.api_key:
            if not API_FALLBACK_AVAILABLE:
                print("❌ Error: API fallback not available. Install openai package: pip install openai")
                return 1
            
            filter_thinking = not args.no_filter_thinking
            api_fallback_function = create_api_fallback_function(
                api_key=args.api_key,
                base_url=args.api_base_url,
                model=args.api_model,
                prompt=args.api_prompt,
                filter_thinking=filter_thinking
            )
            print(f"✅ API fallback configured with model: {args.api_model}")
        else:
            print("ℹ️  No API key provided - only confidence filtering will be applied")
        
        # Process JSON files
        results = process_json_files_for_api_fallback(
            json_dir=args.json_dir,
            image_dir=args.image_dir,
            output_dir=output_dir,
            api_fallback_function=api_fallback_function,
            confidence_threshold=args.confidence_threshold,
            image_extensions=args.extensions,
            overwrite_existing=args.overwrite
        )
        
        # Print summary
        print("\n" + "="*50)
        print("📊 PROCESSING SUMMARY")
        print("="*50)
        print(f"Total JSON files: {results['total_files']}")
        print(f"Files needing API fallback: {results['needs_api_fallback']}")
        print(f"API processing successful: {results['api_success']}")
        print(f"API processing failed: {results['api_failed']}")
        print(f"Confidence filtering applied: {results['confidence_filtered']}")
        print(f"Skipped (no matching image): {results['skipped_no_image']}")
        print(f"Skipped (existing files): {results['skipped_existing']}")
        print(f"Confidence threshold: {args.confidence_threshold}")
        print(f"Output directory: {output_dir}")
        
        if results['processed_files']:
            print(f"\n✅ Successfully processed {len(results['processed_files'])} files:")
            for item in results['processed_files'][:5]:  # Show first 5
                if item['processing_type'] == 'api_fallback':
                    print(f"  - {Path(item['json_file']).name} → {Path(item['output_file']).name} "
                          f"(API: {item.get('api_regions', 0)} regions)")
                else:
                    print(f"  - {Path(item['json_file']).name} → {Path(item['output_file']).name} "
                          f"(Filtered: {item.get('original_regions', 0)} → {item.get('filtered_regions', 0)} regions)")
            if len(results['processed_files']) > 5:
                print(f"  ... and {len(results['processed_files']) - 5} more")
        
        if results['failed_files']:
            print(f"\n❌ Failed to process {len(results['failed_files'])} files:")
            for item in results['failed_files'][:5]:  # Show first 5
                print(f"  - {Path(item['json_file']).name}: {item.get('error', 'Unknown error')}")
            if len(results['failed_files']) > 5:
                print(f"  ... and {len(results['failed_files']) - 5} more")
        
        if results['no_image_files']:
            print(f"\n⚠️  No matching images found for {len(results['no_image_files'])} JSON files:")
            for item in results['no_image_files'][:5]:  # Show first 5
                print(f"  - {Path(item).name}")
            if len(results['no_image_files']) > 5:
                print(f"  ... and {len(results['no_image_files']) - 5} more")
        
        print("\n✅ Processing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
