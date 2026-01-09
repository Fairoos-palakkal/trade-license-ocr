"""
OCR Pipeline - Extract text from detected regions
This module handles the complete OCR pipeline from YOLO detections to extracted text
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, List, Any, Tuple
import easyocr

# ============================================================================
# GLOBAL OCR READER
# ============================================================================

# Initialize EasyOCR reader (loaded once at startup)
_reader = None


def get_ocr_reader() -> easyocr.Reader:
    """
    Get or initialize the EasyOCR reader.
    
    EasyOCR supports multiple languages. For Abu Dhabi licenses:
    - English (en) - For most text
    - Arabic (ar) - For Arabic text
    
    Returns:
        easyocr.Reader: Initialized OCR reader
    """
    global _reader
    
    if _reader is None:
        print("🔤 Initializing EasyOCR reader (English + Arabic)...")
        print("   ⏳ This takes 10-20 seconds on first run...")
        
        # Initialize with English and Arabic
        # gpu=True uses M1 GPU for faster processing
        _reader = easyocr.Reader(
            ['en', 'ar'],  # Languages
            gpu=True,      # Use GPU if available (M1 Neural Engine)
            verbose=False  # Don't print detailed logs
        )
        
        print("   ✅ OCR reader initialized!")
    
    return _reader


# ============================================================================
# STEP 1: CROP IMAGE REGIONS
# ============================================================================

def crop_image_region(
    image: np.ndarray,
    bbox: Dict[str, int],
    padding: int = 5
) -> np.ndarray:
    """
    Crop a region from the image using bounding box coordinates.
    
    Args:
        image: Full image as numpy array (from cv2.imread)
        bbox: Dictionary with keys 'x1', 'y1', 'x2', 'y2'
        padding: Extra pixels to add around the crop (helps with OCR accuracy)
        
    Returns:
        Cropped image as numpy array
        
    Example:
        bbox = {'x1': 50, 'y1': 100, 'x2': 250, 'y2': 130}
        cropped = crop_image_region(image, bbox, padding=5)
    """
    # Extract coordinates
    x1 = int(bbox['x1'])
    y1 = int(bbox['y1'])
    x2 = int(bbox['x2'])
    y2 = int(bbox['y2'])
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Add padding (but keep within image boundaries)
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(img_width, x2 + padding)
    y2_padded = min(img_height, y2 + padding)
    
    # Crop using numpy array slicing
    # Format: image[y1:y2, x1:x2]
    cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
    
    return cropped


# ============================================================================
# STEP 2: PREPROCESS CROPPED IMAGES
# ============================================================================

def preprocess_for_ocr(cropped_image: np.ndarray) -> np.ndarray:
    """
    Preprocess a cropped image to improve OCR accuracy.
    
    Steps:
    1. Convert to grayscale
    2. Resize to optimal height (32-48 pixels)
    3. Increase contrast
    4. Denoise
    5. Apply adaptive thresholding
    
    Args:
        cropped_image: Cropped region as numpy array
        
    Returns:
        Preprocessed image ready for OCR
    """
    # STEP 1: Convert to grayscale
    # OCR works better with grayscale images
    if len(cropped_image.shape) == 3:  # If color image
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_image
    
    # STEP 2: Resize to optimal height
    # OCR engines work best with text height around 32-48 pixels
    height, width = gray.shape
    target_height = 48
    
    if height < target_height:
        # Scale up small text
        scale_factor = target_height / height
        new_width = int(width * scale_factor)
        gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    elif height > target_height * 2:
        # Scale down very large text
        scale_factor = target_height / height
        new_width = int(width * scale_factor)
        gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_AREA)
    
    # STEP 3: Denoise
    # Remove small artifacts and noise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # STEP 4: Increase contrast
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)
    
    # STEP 5: Adaptive thresholding (optional, helps with varied lighting)
    # This creates pure black text on white background
    # Comment out if it makes results worse for your images
    binary = cv2.adaptiveThreshold(
        contrast_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # Constant subtracted from mean
    )
    
    return binary  # Return the best preprocessed version


# ============================================================================
# STEP 3: EXTRACT TEXT FROM PREPROCESSED IMAGE
# ============================================================================

def extract_text_from_image(preprocessed_image: np.ndarray) -> Tuple[str, float]:
    """
    Extract text from a preprocessed image using EasyOCR.
    
    Args:
        preprocessed_image: Preprocessed image as numpy array
        
    Returns:
        Tuple of (extracted_text, confidence_score)
        
    Example:
        text, confidence = extract_text_from_image(image)
        # text = "CN-2883802"
        # confidence = 0.95
    """
    # Get OCR reader
    reader = get_ocr_reader()
    
    # Run OCR
    # allowlist: Characters we expect (helps accuracy)
    # paragraph: False returns individual text blocks
    results = reader.readtext(
        preprocessed_image,
        detail=1,  # Return detailed results with confidence
        paragraph=False  # Don't combine into paragraphs
    )
    
    # Process results
    if not results or len(results) == 0:
        return "", 0.0
    
    # EasyOCR returns: [(bbox, text, confidence), ...]
    # We just need text and confidence
    
    # Combine all detected text (in case text is split into multiple parts)
    texts = []
    confidences = []
    
    for (bbox, text, conf) in results:
        if text.strip():  # Only include non-empty text
            texts.append(text.strip())
            confidences.append(conf)
    
    # Combine texts with space
    combined_text = " ".join(texts)
    
    # Average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return combined_text, avg_confidence


# ============================================================================
# STEP 4: COMPLETE PIPELINE - PROCESS ALL DETECTIONS
# ============================================================================

def process_detections(
    image_path: str,
    detections: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Complete OCR pipeline: Process all YOLO detections and extract text.
    
    This is the main function that combines all steps:
    1. Load image
    2. For each detection:
       a. Crop the region
       b. Preprocess the crop
       c. Extract text
    3. Return results mapped by field name
    
    Args:
        image_path: Path to the original image
        detections: List of YOLO detections from ocr_service.run_inference()
                   Each detection has: class_name, confidence, bbox
        
    Returns:
        Dictionary mapping field names to extracted data:
        {
            'license_number': {
                'text': 'CN-2883802',
                'detection_confidence': 0.95,
                'ocr_confidence': 0.92,
                'bbox': {'x1': 50, 'y1': 100, 'x2': 250, 'y2': 130}
            },
            'company_name': { ... }
        }
    """
    print("\n" + "=" * 60)
    print("🔬 STARTING OCR PIPELINE")
    print("=" * 60)
    
    # STEP 1: Load the full image
    print(f"📂 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f"   ✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # STEP 2: Process each detection
    results = {}
    
    print(f"\n📊 Processing {len(detections)} detections...")
    
    for i, detection in enumerate(detections, 1):
        class_name = detection['class_name']
        bbox = detection['bbox']
        det_confidence = detection['confidence']
        
        print(f"\n   Detection {i}/{len(detections)}: {class_name}")
        print(f"      Detection confidence: {det_confidence:.2%}")
        
        # STEP 2a: Crop the region
        print(f"      ✂️  Cropping region: {bbox}")
        cropped = crop_image_region(image, bbox, padding=5)
        
        if cropped.size == 0:
            print(f"      ⚠️  Empty crop, skipping")
            continue
        
        print(f"      ✅ Cropped: {cropped.shape[1]}x{cropped.shape[0]} pixels")
        
        # STEP 2b: Preprocess
        print(f"      🔧 Preprocessing...")
        preprocessed = preprocess_for_ocr(cropped)
        print(f"      ✅ Preprocessed: {preprocessed.shape[1]}x{preprocessed.shape[0]} pixels")
        
        # STEP 2c: Extract text
        print(f"      🔤 Extracting text...")
        text, ocr_confidence = extract_text_from_image(preprocessed)
        
        if text:
            print(f"      ✅ Extracted: '{text}' (OCR confidence: {ocr_confidence:.2%})")
        else:
            print(f"      ⚠️  No text detected")
        
        # STEP 2d: Store result
        results[class_name] = {
            'text': text,
            'detection_confidence': round(det_confidence, 3),
            'ocr_confidence': round(ocr_confidence, 3),
            'bbox': bbox
        }
    
    print("\n" + "=" * 60)
    print(f"✅ PIPELINE COMPLETE: Extracted {len(results)} fields")
    print("=" * 60 + "\n")
    
    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_debug_crops(
    image_path: str,
    detections: List[Dict[str, Any]],
    output_dir: str = "uploads/debug_crops"
) -> None:
    """
    Save cropped and preprocessed images for debugging.
    
    Useful to visually inspect what's being sent to OCR.
    
    Args:
        image_path: Path to original image
        detections: List of YOLO detections
        output_dir: Where to save debug images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    
    for i, detection in enumerate(detections):
        class_name = detection['class_name']
        bbox = detection['bbox']
        
        # Crop
        cropped = crop_image_region(image, bbox, padding=5)
        
        # Preprocess
        preprocessed = preprocess_for_ocr(cropped)
        
        # Save both versions
        crop_path = os.path.join(output_dir, f"{i}_{class_name}_cropped.jpg")
        prep_path = os.path.join(output_dir, f"{i}_{class_name}_preprocessed.jpg")
        
        cv2.imwrite(crop_path, cropped)
        cv2.imwrite(prep_path, preprocessed)
    
    print(f"💾 Debug crops saved to: {output_dir}")