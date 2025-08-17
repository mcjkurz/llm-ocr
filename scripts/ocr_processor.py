#!/usr/bin/env python3
"""
OCR Processor - Extract text from images using PaddleOCR with single worker process support
"""

import os
import json
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from tqdm import tqdm
import queue
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", message=".*ccache.*")
warnings.filterwarnings("ignore", message=".*MKL-DNN.*")
warnings.filterwarnings("ignore", message=".*PaddlePaddle.*")

# Suppress urllib3 SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

# Set environment variables to suppress PaddlePaddle verbose output
os.environ['GLOG_minloglevel'] = '2'  # Suppress INFO and WARNING logs
os.environ['GLOG_v'] = '0'  # Minimal verbosity
os.environ['FLAGS_logtostderr'] = '0'  # Don't log to stderr
os.environ['FLAGS_stderrthreshold'] = '3'  # Only fatal errors to stderr

# Suppress PaddleX model download and creation verbose output
os.environ['PADDLEX_SILENT'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Ensure tqdm can still display progress bars
os.environ['TQDM_DISABLE'] = '0'

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False



logger = logging.getLogger(__name__)








def _process_single_image(image_path: str, output_dir: str, 
                         ocr_instance: 'PaddleOCR') -> Dict[str, Any]:
    """Process a single image using OCR and save results to JSON file"""
    try:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the provided OCR instance
        result = ocr_instance.predict(str(image_path))
        
        # Generate JSON filename
        json_filename = f"{image_path.stem}.json"
        json_path = output_dir / json_filename
        
        # Create JSON structure
        json_data = {
            "input_path": str(image_path),
            "timestamp": time.time(),
            "total_regions": 0,
            "regions": []
        }
        
        if result and result[0]:
            # Extract text regions from PaddleOCR result
            page_result = result[0]
            text_lines = page_result['rec_texts'] if isinstance(page_result, dict) else page_result.rec_texts
            dt_polys = page_result.get('dt_polys', []) if isinstance(page_result, dict) else getattr(page_result, 'dt_polys', [])
            rec_scores = page_result.get('rec_scores', []) if isinstance(page_result, dict) else getattr(page_result, 'rec_scores', [])
            
            # Create regions with bbox and confidence info
            text_regions = []
            for i, text in enumerate(text_lines):
                bbox = dt_polys[i] if i < len(dt_polys) else None
                confidence = rec_scores[i] if i < len(rec_scores) else None
                
                text_regions.append({
                    "region_id": i,
                    "text": text,
                    "bbox": bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                    "confidence": float(confidence) if confidence is not None else None
                })
            
            json_data["total_regions"] = len(text_regions)
            json_data["regions"] = text_regions
        
        # Save JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        return {
            "image_path": str(image_path),
            "json_path": str(json_path),
            "success": True,
            "regions_count": json_data["total_regions"],
            "error": None
        }
        
    except Exception as e:
        return {
            "image_path": str(image_path),
            "json_path": None,
            "success": False,
            "regions_count": 0,
            "error": str(e)
        }



def _worker_process_function(input_queue: mp.Queue, output_queue: mp.Queue, 
                            lang: str):
    """Worker process functioh"""
    try:
        # Apply same warning suppressions in worker process
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
        warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
        warnings.filterwarnings("ignore", message=".*ccache.*")
        warnings.filterwarnings("ignore", message=".*MKL-DNN.*")
        warnings.filterwarnings("ignore", message=".*PaddlePaddle.*")
        
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
        
        # Set environment variables in worker process too
        import os
        os.environ['GLOG_minloglevel'] = '2'
        os.environ['GLOG_v'] = '0'
        os.environ['FLAGS_logtostderr'] = '0'
        os.environ['FLAGS_stderrthreshold'] = '3'
        os.environ['PADDLEX_SILENT'] = '1'
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
        # Initialize PaddleOCR in worker process
        paddle_ocr = PaddleOCR(lang=lang, device="cpu")
        
        while True:
            try:
                # Get work item from queue with timeout
                work_item = input_queue.get(timeout=1)
                if work_item is None:  # Poison pill to stop worker
                    break
                
                image_path, output_dir = work_item
                
                # Process the image
                result = _process_single_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    ocr_instance=paddle_ocr
                )
                
                # Send result back
                output_queue.put(result)
                
            except queue.Empty:
                # No work available, continue checking
                continue
            except Exception as e:
                # If individual image fails, send error result
                error_result = {
                    "image_path": work_item[0] if work_item else "unknown",
                    "json_path": None,
                    "success": False,
                    "regions_count": 0,
                    "error": f"Worker process error: {str(e)}"
                }
                output_queue.put(error_result)
                
    except Exception as e:
        # Worker initialization failed
        error_result = {
            "image_path": "worker_init",
            "json_path": None,
            "success": False,
            "regions_count": 0,
            "error": f"Worker initialization failed: {str(e)}"
        }
        output_queue.put(error_result)


class SimpleWorkerProcess:
    """Manages a single worker process for OCR tasks with robust error handling"""
    
    def __init__(self, lang: str = 'ch', timeout: int = 30):
        """
        Initialize worker process manager
        
        Args:
            lang: Language code for PaddleOCR
            timeout: Timeout in seconds for each image processing (default: 30)
        """
        self.lang = lang
        self.timeout = timeout
        self.process = None
        self.input_queue = None
        self.output_queue = None
        self.current_image = None
        self.start_time = None
    
    def _start_worker(self):
        """Start the worker process"""
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        
        self.process = mp.Process(
            target=_worker_process_function,
            args=(self.input_queue, self.output_queue, self.lang)
        )
        self.process.start()
        # logger.info(f"Started worker process {self.process.pid}")
    
    def _stop_worker(self):
        """Stop the worker process"""
        if self.process and self.process.is_alive():
            # Send poison pill
            self.input_queue.put(None)
            
            # Wait for graceful shutdown
            self.process.join(timeout=5)
            
            # Force kill if still alive
            if self.process.is_alive():
                logger.warning(f"Force killing worker process {self.process.pid}")
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
        
        self.process = None
        self.input_queue = None
        self.output_queue = None
    
    def _restart_worker(self):
        """Restart the worker process"""
        # logger.info("Restarting worker process...")
        self._stop_worker()
        self._start_worker()
    
    def _run_worker_for_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single image with the worker process"""
        # Start worker if not already running
        if not self.process or not self.process.is_alive():
            self._start_worker()
        
        try:
            self.current_image = image_path
            self.start_time = time.time()
            
            # Send work to worker
            self.input_queue.put((image_path, output_dir))
            
            # Wait for result with timeout
            try:
                result = self.output_queue.get(timeout=self.timeout)
                return result
            except queue.Empty:
                # Timeout - restart worker
                logger.warning(f"Worker timeout for {image_path}, restarting...")
                self._restart_worker()
                return {
                    "image_path": image_path,
                    "json_path": None,
                    "success": False,
                    "regions_count": 0,
                    "error": f"Worker process timeout ({self.timeout}s)"
                }
                
        except Exception as e:
            logger.error(f"Error communicating with worker for {image_path}: {e}")
            self._restart_worker()
            return {
                "image_path": image_path,
                "json_path": None,
                "success": False,
                "regions_count": 0,
                "error": f"Worker communication error: {e}"
            }
        finally:
            self.current_image = None
            self.start_time = None
    
    def process_images(self, image_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process a list of images sequentially with worker process"""
        results = []
        
        try:
            with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
                for image_path in image_paths:
                    try:
                        # Process with worker
                        result = self._run_worker_for_image(image_path, output_dir)
                        results.append(result)
                        
                    except Exception as e:
                        # Final catch-all error handling
                        error_result = {
                            "image_path": image_path,
                            "json_path": None,
                            "success": False,
                            "regions_count": 0,
                            "error": f"Unexpected error: {e}"
                        }
                        results.append(error_result)
                    
                    pbar.update(1)
        finally:
            # Always clean up worker process
            self._stop_worker()
        
        return results


class OCRProcessor:
    """OCR processor with PaddleOCR and single worker process support"""
    
    def __init__(self, lang: str = 'ch', use_worker_process: bool = True, 
                 worker_timeout: int = 30):
        """
        Initialize OCR processor
        
        Args:
            lang: Language code for PaddleOCR ('ch' for Chinese, 'en' for English)
            use_worker_process: Whether to use separate worker process (recommended: True)
            worker_timeout: Timeout in seconds for worker process per image (default: 30)
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR is not installed. Install it with: pip install paddleocr paddlepaddle")
        
        self.lang = lang
        self.use_worker_process = use_worker_process
        self.worker_timeout = worker_timeout
        
        # Initialize either worker process manager or main process OCR
        if use_worker_process:
            self.worker = SimpleWorkerProcess(
                lang=lang,
                timeout=worker_timeout
            )
            self.paddle_ocr = None
        else:
            self.paddle_ocr = PaddleOCR(
                lang=lang,
                use_textline_orientation=True,
                device="cpu"
            )
            self.worker = None
    
    def process_single_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a single image and save JSON result
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save JSON output
            
        Returns:
            Dictionary containing processing results
        """
        if self.use_worker_process:
            # Use worker process for processing
            result = self.worker._run_worker_for_image(image_path, output_dir)
        else:
            # Use main process OCR instance
            result = _process_single_image(
                image_path=image_path,
                output_dir=output_dir,
                ocr_instance=self.paddle_ocr
            )
        
        # Log errors if processing failed
        if not result["success"]:
            logger.error(f"Error processing {image_path}: {result['error']}")
        
        return result
    
    def process_images(self, image_paths: List[str], output_dir: str) -> Dict[str, Any]:
        """
        Process multiple images with single worker process or main process
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save JSON outputs
            
        Returns:
            Dictionary containing processing summary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_images = len(image_paths)
        # logger.info(f"Processing {total_images} images")
        # logger.info(f"Worker process: {self.use_worker_process}")
        # if self.use_worker_process:
        #     logger.info(f"Worker timeout: {self.worker_timeout} seconds per image")
        
        results = []
        start_time = time.time()
        
        if self.use_worker_process:
            # Use worker process for all images
            results = self.worker.process_images(image_paths, str(output_dir))
        else:
            # Sequential processing in main process
            with tqdm(total=total_images, desc="Processing images", 
                      unit="img", ncols=100) as pbar:
                for img_path in image_paths:
                    result = self.process_single_image(img_path, str(output_dir))
                    results.append(result)
                    pbar.update(1)
        
        # Calculate summary
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        total_regions = sum(r["regions_count"] for r in successful)
        elapsed_time = time.time() - start_time
        
        summary = {
            "total_images": total_images,
            "successful": len(successful),
            "failed": len(failed),
            "total_regions": total_regions,
            "elapsed_time": elapsed_time,
            "images_per_second": total_images / elapsed_time if elapsed_time > 0 else 0,
            "output_directory": str(output_dir),
            "failed_images": [r["image_path"] for r in failed]
        }
        
        # logger.info(f"Processing complete: {summary['successful']}/{total_images} successful")
        # logger.info(f"Total regions extracted: {total_regions}")
        # logger.info(f"Time elapsed: {elapsed_time:.2f}s ({summary['images_per_second']:.2f} img/s)")
        
        # Only log failures as errors
        if failed:
            logger.error(f"Failed images: {len(failed)}")
            for result in failed:
                logger.error(f"  {result['image_path']}: {result['error']}")
        
        return summary


def process_images_from_directory(
    image_dir: str,
    output_dir: str,
    image_extensions: List[str] = None,
    use_worker_process: bool = True,
    lang: str = 'ch',
    worker_timeout: int = 30
) -> Dict[str, Any]:
    """
    Process all images in a directory
    
    Args:
        image_dir: Directory containing image files
        output_dir: Directory to save JSON outputs
        image_extensions: List of image extensions to process (default: ['.png', '.jpg', '.jpeg'])
        use_worker_process: Whether to use separate worker process (default: True)
        lang: Language code for PaddleOCR
        worker_timeout: Timeout in seconds for worker process per image (default: 30)
        
    Returns:
        Dictionary containing processing summary
    """
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg']
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Find all image files
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
    
    image_paths = [str(p) for p in sorted(image_paths)]
    
    if not image_paths:
        raise ValueError(f"No image files found in {image_dir} with extensions {image_extensions}")
    
    # logger.info(f"Found {len(image_paths)} image files in {image_dir}")
    
    # Initialize processor and process images
    processor = OCRProcessor(lang=lang, use_worker_process=use_worker_process, 
                           worker_timeout=worker_timeout)
    
    return processor.process_images(image_paths, output_dir)


def main():
    """Command line interface for OCR processing"""
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from images using PaddleOCR")
    parser.add_argument("image_dir", help="Directory containing image files")
    parser.add_argument("output_dir", help="Directory to save JSON outputs")
    parser.add_argument("--no-worker-process", action="store_true",
                        help="Use main process instead of worker process (less robust)")
    parser.add_argument("--worker-timeout", type=int, default=30,
                        help="Timeout in seconds for worker process per image (default: 30)")
    parser.add_argument("--lang", default="ch",
                        help="Language code for PaddleOCR (default: ch)")
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"],
                        help="Image file extensions to process (default: .png .jpg .jpeg)")

    
    args = parser.parse_args()
    
    # Setup logging - only show errors and critical messages
    logging.basicConfig(level=logging.ERROR,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Also suppress PaddleX/PaddlePaddle specific loggers
    logging.getLogger('paddlex').setLevel(logging.ERROR)
    logging.getLogger('paddle').setLevel(logging.ERROR)
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    
    try:
        use_worker_process = not args.no_worker_process
        summary = process_images_from_directory(
            args.image_dir,
            args.output_dir,
            args.extensions,
            use_worker_process,
            args.lang,
            args.worker_timeout
        )
        
        print("\n" + "="*50)
        print("📊 PROCESSING SUMMARY")
        print("="*50)
        print(f"Total images: {summary['total_images']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total text regions: {summary['total_regions']}")
        print(f"Processing time: {summary['elapsed_time']:.2f}s")
        print(f"Speed: {summary['images_per_second']:.2f} images/second")
        print(f"Output directory: {summary['output_directory']}")
        
        if summary['failed_images']:
            print(f"\n❌ Failed images:")
            for img in summary['failed_images']:
                print(f"  - {img}")
        
        print("\n✅ Processing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    mp.freeze_support()
    exit(main())
