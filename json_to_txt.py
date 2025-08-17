#!/usr/bin/env python3
"""
JSON to TXT Converter - Convert OCR JSON results to text with configurable ordering
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class JSONToTxtConverter:
    """Convert OCR JSON results to text with configurable box ordering"""
    
    def __init__(self, box_order: str = "top-bottom", min_line_length: int = 0):
        """
        Initialize converter
        
        Args:
            box_order: Box ordering strategy ("top-bottom", "left-right", "right-left", "confidence")
            min_line_length: Minimum line length to include (default: 0, includes all lines)
        """
        valid_orders = ["top-bottom", "left-right", "right-left", "confidence"]
        if box_order not in valid_orders:
            raise ValueError(f"Invalid box_order: {box_order}. Must be one of {valid_orders}")
        
        self.box_order = box_order
        self.min_line_length = min_line_length
        logger.info(f"Initialized JSON to TXT converter with box order: {box_order}, min line length: {min_line_length}")
    
    def _get_bbox_coords(self, region: Dict[str, Any]) -> Tuple[float, float]:
        """
        Extract x,y coordinates from bbox
        
        Args:
            region: Text region dictionary with bbox information
            
        Returns:
            Tuple of (x, y) coordinates
        """
        bbox = region.get("bbox")
        if bbox is None or (isinstance(bbox, list) and len(bbox) == 0):
            # For fallback mode (no bbox coordinates), use region_id as ordering
            return float(region.get("region_id", 0)), 0.0
        
        # Handle different bbox formats
        if isinstance(bbox, list) and len(bbox) > 0:
            # If it's a polygon (list of [x,y] coordinates), get the first point (top-left)
            if isinstance(bbox[0], list) and len(bbox[0]) >= 2:
                return float(bbox[0][0]), float(bbox[0][1])
            # If it's a simple [x1,y1,x2,y2] format
            elif len(bbox) >= 2:
                return float(bbox[0]), float(bbox[1])
        
        return 0.0, 0.0
    
    def _sort_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort text regions based on the specified box_order
        
        Args:
            regions: List of text regions with bbox information
            
        Returns:
            Sorted list of text regions
        """
        if not regions:
            return regions
        
        if self.box_order == "top-bottom":
            # Sort by y-coordinate (top to bottom)
            return sorted(regions, key=lambda region: self._get_bbox_coords(region)[1])
        
        elif self.box_order == "left-right":
            # Sort by x-coordinate (left to right)
            return sorted(regions, key=lambda region: self._get_bbox_coords(region)[0])
        
        elif self.box_order == "right-left":
            # Sort by x-coordinate, descending (right to left)
            return sorted(regions, key=lambda region: self._get_bbox_coords(region)[0], reverse=True)
        
        elif self.box_order == "confidence":
            # Sort by confidence score, descending (highest confidence first)
            return sorted(regions, key=lambda region: region.get("confidence", 0.0), reverse=True)
        
        else:
            # Default: return as-is
            return regions
    
    def convert_json_to_txt(self, json_path: str) -> str:
        """
        Convert a single JSON file to text
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Extracted text as a string
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            regions = data.get("regions", [])
            
            # Sort regions according to box order
            sorted_regions = self._sort_regions(regions)
            
            # Extract text lines
            text_lines = []
            for region in sorted_regions:
                text = region.get("text", "").strip()
                # Filter by minimum line length and non-empty text
                if text and len(text) >= self.min_line_length:
                    text_lines.append(text)
            
            return '\n'.join(text_lines)
            
        except Exception as e:
            logger.error(f"Error processing JSON file {json_path}: {e}")
            raise
    
    def convert_json_files_to_txt(self, json_paths: List[str], output_file: Optional[str] = None) -> str:
        """
        Convert multiple JSON files to a single text file
        
        Args:
            json_paths: List of JSON file paths
            output_file: Optional output text file path
            
        Returns:
            Combined text from all JSON files
        """
        all_text = []
        
        for i, json_path in enumerate(json_paths, 1):
            try:
                text = self.convert_json_to_txt(json_path)
                if text.strip():  # Only include non-empty text
                    # Add page separator with page number
                    page_separator = f"---- page {i} ----"
                    if i > 1:  # Add separator before content (except for first page)
                        all_text.append(page_separator)
                    all_text.append(text)
                    logger.info(f"Processed {json_path}: {len(text)} characters")
                else:
                    logger.warning(f"No text extracted from {json_path}")
            
            except Exception as e:
                logger.error(f"Failed to process {json_path}: {e}")
                continue
        
        # Combine all text with newlines
        combined_text = '\n\n'.join(all_text)
        
        # Save to file if output path specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(combined_text)
            
            logger.info(f"Saved combined text to: {output_path}")
            logger.info(f"Total length: {len(combined_text)} characters")
        
        return combined_text
    
    def convert_directory_to_txt(self, json_dir: str, output_file: str) -> str:
        """
        Convert all JSON files in a directory to a single text file
        
        Args:
            json_dir: Directory containing JSON files
            output_file: Output text file path
            
        Returns:
            Combined text from all JSON files
        """
        json_dir = Path(json_dir)
        
        if not json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")
        
        # Find all JSON files
        json_files = list(json_dir.glob("*.json"))
        json_files = sorted(json_files)  # Sort for consistent ordering
        
        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")
        
        logger.info(f"Found {len(json_files)} JSON files in {json_dir}")
        
        json_paths = [str(f) for f in json_files]
        
        return self.convert_json_files_to_txt(json_paths, output_file)


def main():
    """Command line interface for JSON to TXT conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert OCR JSON results to text")
    parser.add_argument("input", help="JSON file or directory containing JSON files")
    parser.add_argument("output", help="Output text file path")
    parser.add_argument("--box-order", choices=["top-bottom", "left-right", "right-left", "confidence"],
                        default="top-bottom", help="Box ordering strategy (default: top-bottom)")
    parser.add_argument("--min-line-length", type=int, default=0,
                        help="Minimum line length to include (default: 0)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        converter = JSONToTxtConverter(box_order=args.box_order, min_line_length=args.min_line_length)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single JSON file
            if input_path.suffix.lower() != '.json':
                raise ValueError(f"Input file must be a JSON file: {input_path}")
            
            text = converter.convert_json_to_txt(str(input_path))
            
            # Save to output file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"✅ Converted {input_path} to {output_path}")
            print(f"📊 Text length: {len(text)} characters")
        
        elif input_path.is_dir():
            # Directory of JSON files
            text = converter.convert_directory_to_txt(str(input_path), args.output)
            
            print(f"✅ Converted JSON files from {input_path} to {args.output}")
            print(f"📊 Combined text length: {len(text)} characters")
        
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
