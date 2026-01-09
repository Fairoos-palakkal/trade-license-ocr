"""
API Routes - OCR Endpoints
Defines all API endpoints for the Trade License OCR system
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import time

from app.models.schemas import OCRResponse, ErrorResponse, TradeLicenseData, ValidationInfo
from app.services.ocr_service import run_inference
from app.services.ocr_pipeline import process_detections
from app.services.parser_service import process_and_structure
from app.utils.file_handler import save_upload_file, cleanup_file, check_file_size


# ============================================================================
# CREATE ROUTER
# ============================================================================

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)


# ============================================================================
# POST /ocr/trade-license - Main OCR Endpoint
# ============================================================================

@router.post(
    "/trade-license",
    response_model=OCRResponse,
    summary="Process Abu Dhabi Trade License",
    description="""
    Upload an image or PDF of an Abu Dhabi Trade License and receive structured data.
    
    **Supported formats:** JPG, JPEG, PNG, PDF
    
    **Maximum file size:** 10MB
    
    **Processing steps:**
    1. Upload file validation
    2. YOLO object detection (find fields)
    3. OCR text extraction (read text)
    4. Post-processing (clean & structure)
    5. Validation
    
    **Returns:** Structured JSON with license information
    """,
    responses={
        200: {
            "description": "Successfully processed trade license",
            "model": OCRResponse
        },
        400: {
            "description": "Invalid file or bad request",
            "model": ErrorResponse
        },
        500: {
            "description": "Server error during processing",
            "model": ErrorResponse
        }
    }
)
async def process_trade_license(
    file: UploadFile = File(
        ...,
        description="Trade license image or PDF file",
        example="trade_license.jpg"
    )
) -> OCRResponse:
    """
    Process a trade license image/PDF and extract structured data.
    
    This endpoint orchestrates the complete OCR pipeline:
    1. Validates and saves the uploaded file
    2. Runs YOLO detection to find fields
    3. Extracts text using OCR
    4. Parses and structures the data
    5. Validates the results
    6. Returns clean JSON response
    
    Args:
        file: Uploaded trade license file (image or PDF)
        
    Returns:
        OCRResponse with extracted license data
        
    Raises:
        HTTPException: If processing fails at any step
    """
    file_path = None
    start_time = time.time()
    
    try:
        # ================================================================
        # STEP 1: VALIDATE FILE
        # ================================================================
        print("\n" + "=" * 70)
        print("📄 NEW OCR REQUEST")
        print("=" * 70)
        print(f"📎 File: {file.filename}")
        print(f"📦 Content Type: {file.content_type}")
        
        # Check file size
        is_valid_size, size_error = check_file_size(file)
        if not is_valid_size:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": size_error,
                    "error_type": "FileSizeError"
                }
            )
        
        # ================================================================
        # STEP 2: SAVE UPLOADED FILE
        # ================================================================
        print("\n💾 Saving uploaded file...")
        try:
            file_path = save_upload_file(file)
            print(f"   ✅ File saved: {file_path}")
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": str(e),
                    "error_type": "FileValidationError"
                }
            )
        except IOError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "message": str(e),
                    "error_type": "FileSaveError"
                }
            )
        
        # ================================================================
        # STEP 3: RUN YOLO DETECTION
        # ================================================================
        print("\n🎯 Running YOLO detection...")
        try:
            detections = run_inference(file_path, confidence_threshold=0.25)
            print(f"   ✅ Found {len(detections)} detections")
            
            if len(detections) == 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "success": False,
                        "message": "No fields detected in the image. Please ensure the image is a valid Abu Dhabi Trade License.",
                        "error_type": "NoDetectionsError",
                        "details": {
                            "suggestion": "Try with a clearer image or check if the document is a valid trade license"
                        }
                    }
                )
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "message": f"YOLO detection failed: {str(e)}",
                    "error_type": "DetectionError"
                }
            )
        
        # ================================================================
        # STEP 4: RUN OCR PIPELINE
        # ================================================================
        print("\n🔤 Running OCR pipeline...")
        try:
            ocr_results = process_detections(file_path, detections)
            print("\n🧪 DEBUG ocr_results (AFTER OCR PIPELINE):")
            print(ocr_results)
            print(type(ocr_results))
            print(f"   ✅ Extracted text from {len(ocr_results)} fields")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "message": f"OCR processing failed: {str(e)}",
                    "error_type": "OCRError"
                }
            )
        
        # ================================================================
        # STEP 5: PARSE AND STRUCTURE DATA
        # ================================================================
        print("\n🗂️  Parsing and structuring data...")
        try:
            structured_data = process_and_structure(ocr_results)
            print("   ✅ Data structured successfully")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "message": f"Data parsing failed: {str(e)}",
                    "error_type": "ParsingError"
                }
            )
        
        print("\n🧪 DEBUG structured_data:")
        print(structured_data)
        print(type(structured_data))

        # ================================================================
        # STEP 6: BUILD RESPONSE
        # ================================================================
        processing_time = time.time() - start_time
        
        # Extract validation info
        validation_data = structured_data.pop('validation', {})
        
        # Build response
        response = OCRResponse(
            success=True,
            message="Trade license processed successfully",
            data=TradeLicenseData(**structured_data),
            validation=ValidationInfo(**validation_data) if validation_data else None,
            processing_time=round(processing_time, 2)
        )
        
        print("\n" + "=" * 70)
        print(f"✅ REQUEST COMPLETED in {processing_time:.2f}s")
        print("=" * 70 + "\n")
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions (already formatted)
        raise
    
    except Exception as e:
        # Catch any unexpected errors
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "An unexpected error occurred during processing",
                "error_type": "UnexpectedError",
                "details": {
                    "error": str(e)
                }
            }
        )
    
    finally:
        # ================================================================
        # STEP 7: CLEANUP
        # ================================================================
        # Always clean up uploaded file, even if processing failed
        if file_path:
            cleanup_file(file_path)


# ============================================================================
# GET /ocr/health - Health Check for OCR Service
# ============================================================================

@router.get(
    "/health",
    summary="OCR Service Health Check",
    description="Check if OCR service is operational and models are loaded",
    tags=["Health"]
)
async def ocr_health_check():
    """
    Check OCR service health and model status.
    
    Returns:
        Service status and model information
    """
    try:
        from app.services.ocr_service import get_model_info
        
        model_info = get_model_info()
        
        return {
            "status": "healthy",
            "service": "OCR Service",
            "model_loaded": model_info.get('model_loaded', False),
            "model_info": model_info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "OCR Service",
            "error": str(e)
        }