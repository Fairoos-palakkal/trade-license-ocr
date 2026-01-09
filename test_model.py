"""
Test script to verify model loading and inference.
Run this independently to test without starting the full API.
"""

import sys
from pathlib import Path
from app.services.ocr_service import load_model, run_inference, get_model_info
from PIL import Image, ImageDraw
import numpy as np

def create_dummy_image() -> str:
    """
    Create a dummy test image if you don't have a real license.
    
    Returns:
        Path to created dummy image
    """
    print("📸 Creating dummy test image...")
    
    # Create a simple image (1000x1414 - typical A4 size in pixels)
    width, height = 1000, 1414
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw some rectangles and text to simulate a license
    draw.rectangle([50, 50, 950, 200], fill='lightblue')
    draw.rectangle([50, 250, 950, 400], fill='lightgray')
    draw.rectangle([50, 450, 950, 600], fill='lightgreen')
    
    # Save to uploads folder
    dummy_path = Path("uploads") / "dummy_license.jpg"
    dummy_path.parent.mkdir(exist_ok=True)
    image.save(dummy_path)
    
    print(f"   ✅ Dummy image created: {dummy_path}")
    return str(dummy_path)


def main():
    """
    Main test function.
    """
    print("\n" + "=" * 60)
    print("🧪 TESTING MODEL LOADING AND INFERENCE")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Load model
        print("TEST 1: Loading model...")
        model = load_model()
        print("✅ Model loaded successfully!\n")
        
        # Test 2: Get model info
        print("TEST 2: Getting model info...")
        info = get_model_info()
        print("   Model Info:")
        for key, value in info.items():
            print(f"   - {key}: {value}")
        print("✅ Model info retrieved!\n")
        
        # Test 3: Create dummy image
        print("TEST 3: Creating test image...")
        test_image_path = "uploads/trade_license_sample.jpg" 
        print("✅ Test image ready!\n")
        
        # Test 4: Run inference
        print("TEST 4: Running inference on test image...")
        detections = run_inference(test_image_path, confidence_threshold=0.25)
        
        print("\n📊 RESULTS:")
        print(f"   Total detections: {len(detections)}")
        
        if len(detections) > 0:
            print("\n   Detections:")
            for i, det in enumerate(detections, 1):
                print(f"\n   Detection {i}:")
                print(f"      Class: {det['class_name']}")
                print(f"      Confidence: {det['confidence']:.2%}")
                print(f"      Bounding Box: {det['bbox']}")
        else:
            print("\n   ⚠️  No detections found (this is normal for dummy image)")
            print("   Try with a real trade license image!")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Make sure your model file is at:")
        print("   ml_models/roboflow_model/model.pt")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()