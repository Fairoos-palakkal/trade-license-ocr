"""
OCR Service - Handles model loading and inference
This service manages the Roboflow model for Trade License OCR
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ============================================================================
# GLOBAL MODEL VARIABLE
# ============================================================================

# This will store the loaded model (loaded once at startup)
_model = None


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model() -> YOLO:
    """
    Load the Roboflow model from disk.
    
    This function:
    1. Finds the model file path
    2. Checks if file exists
    3. Loads the .pt file into memory
    4. Returns the model object
    
    Returns:
        YOLO: Loaded model ready for inference
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    global _model
    
    # If model already loaded, return it (don't reload)
    if _model is not None:
        print("✅ Model already loaded, using cached version")
        return _model
    
    # Get the project root directory
    # This file is in: app/services/ocr_service.py
    # Project root is: ../../ (two levels up)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    
    # Build path to model file
    model_path = project_root / "ml_models" / "roboflow_model" / "model.pt"
    
    print(f"🔍 Looking for model at: {model_path}")
    
    # Check if model file exists
    if not model_path.exists():
        error_msg = f"❌ Model file not found at: {model_path}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    print(f"📦 Loading model from: {model_path}")
    
    try:
        # Load the model using Ultralytics YOLO
        _model = YOLO(str(model_path))
        
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(_model)}")
        print(f"   Device: {_model.device}")
        
        return _model
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image_path: str) -> Image.Image:
    """
    Prepare image for model inference.
    
    Steps:
    1. Open image file
    2. Convert to RGB (remove alpha channel if present)
    3. Validate image is readable
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL.Image: Preprocessed image
        
    Raises:
        ValueError: If image cannot be opened
    """
    try:
        # Open image
        image = Image.open(image_path)
        
        # Convert to RGB (handles PNG with transparency, CMYK, etc.)
        if image.mode != 'RGB':
            print(f"   Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        print(f"   ✅ Image loaded: {image.size[0]}x{image.size[1]} pixels")
        
        return image
        
    except Exception as e:
        error_msg = f"❌ Error opening image: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg)


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def run_inference(image_path: str, confidence_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Run OCR inference on an image.
    
    This function:
    1. Loads the model (if not already loaded)
    2. Preprocesses the image
    3. Runs inference
    4. Extracts detected text and bounding boxes
    5. Returns structured results
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence score (0-1)
        
    Returns:
        List of detections, each containing:
        {
            'text': 'detected text',
            'confidence': 0.95,
            'bbox': {
                'x1': 100, 'y1': 50,
                'x2': 300, 'y2': 100
            },
            'class_name': 'license_number'
        }
    """
    print("\n" + "=" * 60)
    print("🔬 STARTING OCR INFERENCE")
    print("=" * 60)
    
    # Step 1: Load model
    print("📦 Step 1: Loading model...")
    model = load_model()
    
    # Step 2: Preprocess image
    print("🖼️  Step 2: Preprocessing image...")
    image = preprocess_image(image_path)
    
    # Step 3: Run inference
    print("🤖 Step 3: Running inference...")
    results = model.predict(
        source=image,
        conf=confidence_threshold,  # Minimum confidence
        verbose=False  # Don't print detailed logs
    )
    
    # Step 4: Extract results
    print("📊 Step 4: Extracting results...")
    detections = []
    
    # Iterate through all detections
    for result in results:
        # Get bounding boxes
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            print("   ⚠️  No detections found")
            continue
        
        print(f"   Found {len(boxes)} detections")
        
        # Process each detection
        for box in boxes:
            # Extract box coordinates (x1, y1, x2, y2)
            coords = box.xyxy[0].cpu().numpy()  # Move to CPU and convert to numpy
            
            # Extract confidence score
            confidence = float(box.conf[0])
            
            # Extract class ID and name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # Get class name from model
            
            # Build detection dictionary
            detection = {
                'class_name': class_name,
                'confidence': round(confidence, 3),
                'bbox': {
                    'x1': int(coords[0]),
                    'y1': int(coords[1]),
                    'x2': int(coords[2]),
                    'y2': int(coords[3])
                }
            }
            
            detections.append(detection)
            
            print(f"   ✅ Detected: {class_name} (confidence: {confidence:.2%})")
    
    print("=" * 60)
    print(f"✅ INFERENCE COMPLETE: {len(detections)} detections")
    print("=" * 60 + "\n")
    
    return detections


# ============================================================================
# INFERENCE WITH OCR TEXT EXTRACTION (if your model includes OCR)
# ============================================================================

def run_inference_with_ocr(image_path: str) -> Dict[str, Any]:
    """
    Run inference and extract text from detected regions.
    
    If your Roboflow model includes OCR (text recognition),
    this function will extract the actual text content.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with detections and extracted text
    """
    # Get detections
    detections = run_inference(image_path)
    
    # If your model has OCR capabilities, text will be in results
    # This depends on how you trained your Roboflow model
    
    result = {
        'total_detections': len(detections),
        'detections': detections,
        'image_path': image_path
    }
    
    return result


# ============================================================================
# HELPER: Get Model Info
# ============================================================================

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Useful for debugging and monitoring.
    
    Returns:
        Dictionary with model information
    """
    model = load_model()
    
    return {
        'model_loaded': _model is not None,
        'model_type': str(type(model)),
        'device': str(model.device),
        'class_names': model.names if hasattr(model, 'names') else []
    }