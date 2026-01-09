"""
Test script for parser service.
Tests field mapping, cleaning, and structuring.
"""

from app.services.ocr_service import run_inference
from app.services.ocr_pipeline import process_detections
from app.services.parser_service import process_and_structure
import json


def test_parser_with_real_data():
    """
    Test complete pipeline: YOLO → OCR → Parsing → Structuring
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING COMPLETE PIPELINE WITH PARSER")
    print("=" * 70 + "\n")
    
    # Use your test image
    image_path = "uploads/dummy_license.jpg"
    
    try:
        # STEP 1: YOLO Detection
        print("🎯 STEP 1: Running YOLO detection...")
        detections = run_inference(image_path, confidence_threshold=0.25)
        print(f"✅ Found {len(detections)} detections\n")
        
        if len(detections) == 0:
            print("⚠️  No detections. Using mock data for testing parser...\n")
            # Create mock OCR results for testing
            ocr_results = create_mock_ocr_results()
        else:
            # STEP 2: OCR Pipeline
            print("🔤 STEP 2: Running OCR pipeline...")
            ocr_results = process_detections(image_path, detections)
            print(f"✅ Extracted text from {len(ocr_results)} fields\n")
        
        # STEP 3: Post-processing and Structuring
        print("🗂️  STEP 3: Post-processing and structuring...")
        structured_data = process_and_structure(ocr_results)
        
        # STEP 4: Display final output
        print("\n" + "=" * 70)
        print("📊 FINAL STRUCTURED OUTPUT")
        print("=" * 70 + "\n")
        
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        
        # Save to file
        output_file = "uploads/final_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Output saved to: {output_file}")
        
        print("\n" + "=" * 70)
        print("✅ COMPLETE PIPELINE TEST SUCCESSFUL!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def create_mock_ocr_results():
    """
    Create mock OCR results for testing parser when no real license is available.
    """
    print("📝 Creating mock OCR results based on sample licenses...\n")
    
    return {
        'license_number': {
            'text': ' CN - 2883802  ',
            'detection_confidence': 0.95,
            'ocr_confidence': 0.92
        },
        'trade_name': {
            'text': 'TIGER  FIRE FOR  SECURITY SYSTEM',
            'detection_confidence': 0.89,
            'ocr_confidence': 0.88
        },
        'establishment_date': {
            'text': '14/10/2019',
            'detection_confidence': 0.87,
            'ocr_confidence': 0.90
        },
        'expiry_date': {
            'text': '25-12-2024',
            'detection_confidence': 0.91,
            'ocr_confidence': 0.93
        },
        'owner': {
            'text': 'NAJLA ANTAR ABDULLA ALJABERI   100 %',
            'detection_confidence': 0.88,
            'ocr_confidence': 0.85
        },
        'manager': {
            'text': 'NAJLA  ANTAR  ABDULLA ALJABERI',
            'detection_confidence': 0.86,
            'ocr_confidence': 0.84
        }
    }


def test_parser_with_multiple_owners():
    """
    Test parser with multiple owners.
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING PARSER WITH MULTIPLE OWNERS")
    print("=" * 70 + "\n")
    
    # Mock data with multiple owners
    ocr_results = {
        'license_number': {
            'text': 'CN-2477716',
            'detection_confidence': 0.96,
            'ocr_confidence': 0.94
        },
        'company_name': {
            'text': 'AL MASSA ARCHITECTURAL GENERAL CONTRACTING',
            'detection_confidence': 0.92,
            'ocr_confidence': 0.89
        },
        'establishment_date': {
            'text': '09.01.2018',
            'detection_confidence': 0.88,
            'ocr_confidence': 0.91
        },
        'expiry_date': {
            'text': '07/01/2024',
            'detection_confidence': 0.90,
            'ocr_confidence': 0.92
        },
        'owner_1': {
            'text': 'MOHAMMED JUMA MOHAMMED SUHAIL ALKAABI 50%',
            'detection_confidence': 0.87,
            'ocr_confidence': 0.83
        },
        'owner_2': {
            'text': 'AHMED HASSAN MOHAMMED 50.0%',
            'detection_confidence': 0.85,
            'ocr_confidence': 0.82
        },
        'manager': {
            'text': 'MOHAMMED JUMA MOHAMMED SUHAIL ALKAABI',
            'detection_confidence': 0.89,
            'ocr_confidence': 0.86
        }
    }
    
    # Process
    structured_data = process_and_structure(ocr_results)
    
    # Display
    print("\n" + "=" * 70)
    print("📊 STRUCTURED OUTPUT (MULTIPLE OWNERS)")
    print("=" * 70 + "\n")
    print(json.dumps(structured_data, indent=2, ensure_ascii=False))
    
    print("\n✅ Multiple owners test complete!\n")


def test_edge_cases():
    """
    Test parser with edge cases and missing fields.
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING EDGE CASES")
    print("=" * 70 + "\n")
    
    # Test 1: Missing fields
    print("Test 1: Missing establishment date and manager")
    ocr_results_1 = {
        'license_number': {'text': 'CN-1234567', 'detection_confidence': 0.9, 'ocr_confidence': 0.88},
        'company_name': {'text': 'TEST COMPANY LLC', 'detection_confidence': 0.85, 'ocr_confidence': 0.82},
        'expiry_date': {'text': '31/12/2025', 'detection_confidence': 0.87, 'ocr_confidence': 0.89},
        'owner': {'text': 'JOHN DOE', 'detection_confidence': 0.80, 'ocr_confidence': 0.78}
    }
    result_1 = process_and_structure(ocr_results_1)
    print(json.dumps(result_1, indent=2, ensure_ascii=False))
    print()
    
    # Test 2: Weird date formats
    print("\nTest 2: Various date formats")
    ocr_results_2 = {
        'license_number': {'text': 'CN-9999999', 'detection_confidence': 0.9, 'ocr_confidence': 0.88},
        'company_name': {'text': 'DATE TEST COMPANY', 'detection_confidence': 0.85, 'ocr_confidence': 0.82},
        'establishment_date': {'text': '1 5 2020', 'detection_confidence': 0.75, 'ocr_confidence': 0.70},
        'expiry_date': {'text': '15.06.2026', 'detection_confidence': 0.80, 'ocr_confidence': 0.78}
    }
    result_2 = process_and_structure(ocr_results_2)
    print(json.dumps(result_2, indent=2, ensure_ascii=False))
    
    print("\n✅ Edge cases test complete!\n")


if __name__ == "__main__":
    # Run all tests
    test_parser_with_real_data()
    test_parser_with_multiple_owners()
    test_edge_cases()
    