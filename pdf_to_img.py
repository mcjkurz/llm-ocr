#!/usr/bin/env python3
"""
PDF to Images Converter - Convert PDF pages to image files
"""

import os
import io
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Generator, Tuple, Optional, List

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _process_page_slice(args) -> List[Tuple[int, str, int]]:
    """
    Worker function to process a slice of PDF pages
    
    Args:
        args: Tuple containing (pdf_data, page_start, page_end, output_dir, format, dpi, crop_margins, progress_counter, lock)
        
    Returns:
        List of tuples (page_number, image_file_path, total_pages)
    """
    pdf_data, page_start, page_end, output_dir, format, dpi, crop_margins, progress_counter, lock = args
    output_dir = Path(output_dir)
    results = []
    
    try:
        # Create document from PDF data in memory
        doc = fitz.open("pdf", pdf_data)
        total_pages = len(doc)
        
        for page_num in range(page_start, min(page_end, total_pages)):
            page = doc.load_page(page_num)
            
            # Convert to image with specified DPI
            scale_factor = dpi / 72.0
            mat = fitz.Matrix(scale_factor, scale_factor)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Scale image to maximum width of 800px while maintaining aspect ratio
            if pil_image.width > 800:
                aspect_ratio = pil_image.height / pil_image.width
                new_width = 800
                new_height = int(new_width * aspect_ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop margins if specified
            if crop_margins:
                top, bottom, left, right = crop_margins
                width, height = pil_image.size
                
                # Calculate crop box (left, top, right, bottom)
                crop_left = int(width * left)
                crop_top = int(height * top)
                crop_right = int(width * (1 - right))
                crop_bottom = int(height * (1 - bottom))
                
                # Ensure crop box is valid
                if crop_left < crop_right and crop_top < crop_bottom:
                    pil_image = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Convert to RGB for maximum compatibility
            if pil_image.mode != 'RGB':
                if pil_image.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                    if pil_image.mode == 'P':
                        pil_image = pil_image.convert('RGBA')
                    if pil_image.mode in ('RGBA', 'LA'):
                        rgb_image.paste(pil_image, mask=pil_image.split()[-1])
                    else:
                        rgb_image.paste(pil_image)
                    pil_image = rgb_image
                else:
                    pil_image = pil_image.convert('RGB')
            
            # Generate output filename
            page_filename = f"page_{page_num + 1:03d}.{format}"
            image_path = output_dir / page_filename
            
            # Save image with format-specific optimization
            save_kwargs = {"optimize": True}
            if format in ['jpg', 'jpeg']:
                save_kwargs["quality"] = 95
                save_kwargs["format"] = "JPEG"
            else:
                save_kwargs["format"] = "PNG"
            
            pil_image.save(image_path, **save_kwargs)
            
            # Verify the file was saved correctly
            if not image_path.exists() or image_path.stat().st_size == 0:
                raise IOError(f"Failed to save image file: {image_path}")
            
            # Update shared progress counter
            with lock:
                progress_counter.value += 1
            
            # Clean up resources
            pix = None
            page = None
            
            results.append((page_num + 1, str(image_path), total_pages))
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error in worker processing pages {page_start}-{page_end}: {e}")
        raise
    
    return results


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    format: str = "png",
    dpi: float = 150.0,
    num_processes: Optional[int] = None,
    crop_margins: Optional[Tuple[float, float, float, float]] = None
) -> Generator[Tuple[int, str, int], None, None]:
    """
    Convert PDF pages to image files with maximum width of 800px using multiprocessing
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save image files
        format: Image format ('png', 'jpg', 'jpeg')
        dpi: DPI for image conversion (higher = better quality but larger files)
        num_processes: Number of worker processes (default: CPU count)
        crop_margins: Optional tuple of (top, bottom, left, right) margin fractions to crop
        
    Note:
        - Images are automatically scaled to maximum width of 800px while maintaining aspect ratio
        - All images are saved in RGB color mode for maximum compatibility
        - Uses multiprocessing for faster conversion of large PDFs
        - Crop margins are specified as fractions (0.0-1.0) of the image dimensions
        
    Yields:
        Tuple of (page_number, image_file_path, total_pages)
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate format
    format = format.lower()
    if format not in ['png', 'jpg', 'jpeg']:
        raise ValueError(f"Unsupported format: {format}. Use 'png', 'jpg', or 'jpeg'")
    
    # Validate crop margins
    if crop_margins is not None:
        if len(crop_margins) != 4:
            raise ValueError("crop_margins must contain exactly 4 values: (top, bottom, left, right)")
        
        top, bottom, left, right = crop_margins
        for margin_name, margin_value in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
            if not 0 <= margin_value <= 1:
                raise ValueError(f"{margin_name} margin must be between 0 and 1, got {margin_value}")
        
        # Check that margins don't overlap
        if top + bottom >= 1:
            raise ValueError(f"Top and bottom margins ({top} + {bottom}) cannot be >= 1")
        if left + right >= 1:
            raise ValueError(f"Left and right margins ({left} + {right}) cannot be >= 1")
    
    # Set number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    logger.info(f"Converting PDF to {format.upper()} images: {pdf_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DPI: {dpi}")
    logger.info(f"Using {num_processes} processes")
    
    try:
        # Read PDF data into memory to pass to worker processes
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        # Get total pages count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        logger.info(f"PDF has {total_pages} pages")
        
        # Calculate page slices for each worker
        pages_per_process = max(1, total_pages // num_processes)
        page_slices = []
        
        for i in range(num_processes):
            start_page = i * pages_per_process
            if i == num_processes - 1:
                # Last process handles remaining pages
                end_page = total_pages
            else:
                end_page = start_page + pages_per_process
            
            if start_page < total_pages:
                page_slices.append((start_page, end_page))
        
        # Create shared progress tracking
        manager = mp.Manager()
        progress_counter = manager.Value('i', 0)
        lock = manager.Lock()
        
        # Prepare arguments for worker processes
        worker_args = []
        for start_page, end_page in page_slices:
            worker_args.append((
                pdf_data, start_page, end_page, str(output_dir), 
                format, dpi, crop_margins, progress_counter, lock
            ))
        
        # Create progress bar
        with tqdm(total=total_pages, desc="Converting PDF to images", 
                  unit="page", ncols=100) as pbar:
            
            # Start multiprocessing pool
            with mp.Pool(processes=num_processes) as pool:
                # Start async processing
                async_results = [pool.apply_async(_process_page_slice, (args,)) for args in worker_args]
                
                # Monitor progress
                last_progress = 0
                while any(not result.ready() for result in async_results):
                    current_progress = progress_counter.value
                    if current_progress > last_progress:
                        pbar.update(current_progress - last_progress)
                        last_progress = current_progress
                    
                    # Small delay to avoid busy waiting
                    time.sleep(0.1)
                
                # Final progress update
                final_progress = progress_counter.value
                if final_progress > last_progress:
                    pbar.update(final_progress - last_progress)
                
                # Collect results from all workers
                all_results = []
                for result in async_results:
                    try:
                        worker_results = result.get(timeout=300)  # 5 minute timeout per worker
                        all_results.extend(worker_results)
                    except Exception as e:
                        logger.error(f"Worker process failed: {e}")
                        raise
        
        # Sort results by page number
        all_results.sort(key=lambda x: x[0])
        
        # Yield results in page order
        for result in all_results:
            yield result
        
        logger.info("PDF to image conversion completed using multiprocessing")
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        raise


def main():
    """Command line interface for PDF to images conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PDF pages to image files using multiprocessing")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("output_dir", help="Directory to save image files")
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png",
                        help="Image format (default: png)")
    parser.add_argument("--dpi", type=float, default=150.0,
                        help="DPI for image conversion (default: 150)")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of worker processes (default: CPU count)")
    parser.add_argument("--crop-margins", type=float, nargs=4, metavar=('TOP', 'BOTTOM', 'LEFT', 'RIGHT'),
                        help="Crop margins as fractions (0.0-1.0): top bottom left right (e.g., 0.1 0.1 0.05 0.05)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Convert crop margins list to tuple if provided
        crop_margins = tuple(args.crop_margins) if args.crop_margins else None
        
        # The progress bar is already shown in pdf_to_images function
        converted_images = list(pdf_to_images(
            args.pdf_path, args.output_dir, args.format, args.dpi, args.processes, crop_margins
        ))
        
        print(f"\n✅ Conversion complete! {len(converted_images)} images saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.freeze_support()
    exit(main())
