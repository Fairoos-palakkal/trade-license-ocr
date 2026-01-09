"""
Test script for OCR pipeline.
Tests the complete flow: YOLO detection → Cropping → OCR → Text extraction
"""

from app.services.ocr_service import run_inference
from app.services.ocr_pipeline import process_detections, save_debug_crops
import json


def test_complete_pipeline():
    """
    Test the complete OCR pipeline.
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING COMPLETE OCR PIPELINE")
    print("=" * 70 + "\n")
    
    # STEP 1: Specify image path
    # Replace this with path to your actual trade license image
    image_path = "uploads/trade_license_sample.jpg"
    
    print(f"📸 Using image: {image_path}")
    print("\n" + "-" * 70)
    
    try:
        # STEP 2: Run YOLO detection
        print("\n🎯 STEP 1: Running YOLO detection...")
        detections = run_inference(image_path, confidence_threshold=0.25)
        
        if len(detections) == 0:
            print("\n⚠️  No detections found!")
            print("💡 Tips:")
            print("   - Make sure you're using a real trade license image")
            print("   - Try lowering confidence_threshold to 0.1")
            print("   - Verify your model is trained correctly")
            return
        
        print(f"✅ Found {len(detections)} detections")
        
        # STEP 3: Run OCR pipeline
        print("\n🔤 STEP 2: Running OCR pipeline...")
        results = process_detections(image_path, detections)
        
        # STEP 4: Display results
        print("\n" + "=" * 70)
        print("📊 EXTRACTION RESULTS")
        print("=" * 70 + "\n")
        
        for field_name, data in results.items():
            print(f"🏷️  {field_name.upper()}")
            print(f"   Text: {data['text']}")
            print(f"   Detection Confidence: {data['detection_confidence']:.2%}")
            print(f"   OCR Confidence: {data['ocr_confidence']:.2%}")
            print(f"   Bounding Box: {data['bbox']}")
            print()
        
        # STEP 5: Save as JSON for inspection
        output_file = "uploads/ocr_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        
        # STEP 6: Save debug crops (optional - helps with debugging)
        print("\n📸 Saving debug crops...")
        save_debug_crops(image_path, detections)
        
        print("\n" + "=" * 70)
        print("✅ TEST COMPLETE!")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Make sure:")
        print("   1. Your model file exists at: ml_models/roboflow_model/model.pt")
        print("   2. Your test image exists at the specified path")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_complete_pipeline()