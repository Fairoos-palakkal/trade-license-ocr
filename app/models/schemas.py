"""
Pydantic Models - API Request/Response Schemas
Defines the structure of data going in and out of the API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# OWNER MODEL
# ============================================================================

class Owner(BaseModel):
    """
    Model for company owner/partner information.
    """
    name: str = Field(..., description="Full name of the owner")
    share_percentage: str = Field(..., description="Ownership percentage (e.g., '50%')")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "NAJLA ANTAR ABDULLA ALJABERI",
                "share_percentage": "100%"
            }
        }


# ============================================================================
# TRADE LICENSE DATA MODEL
# ============================================================================

class TradeLicenseData(BaseModel):
    """
    Model for extracted trade license information.
    This is the core data structure returned by the OCR pipeline.
    """
    license_number: Optional[str] = Field(None, description="Trade license number (e.g., CN-2883802)")
    company_name: Optional[str] = Field(None, description="Company/Trade name")
    establishment_date: Optional[str] = Field(None, description="Date of establishment (DD-MM-YYYY)")
    expiry_date: Optional[str] = Field(None, description="License expiry date (DD-MM-YYYY)")
    manager_name: Optional[str] = Field(None, description="Manager/Authorized signatory name")
    owners: List[Owner] = Field(default_factory=list, description="List of company owners with share percentages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "license_number": "CN-2883802",
                "company_name": "TIGER FIRE FOR SECURITY SYSTEM",
                "establishment_date": "14-10-2019",
                "expiry_date": "25-12-2024",
                "manager_name": "NAJLA ANTAR ABDULLA ALJABERI",
                "owners": [
                    {
                        "name": "NAJLA ANTAR ABDULLA ALJABERI",
                        "share_percentage": "100%"
                    }
                ]
            }
        }


# ============================================================================
# VALIDATION MODEL
# ============================================================================

class ValidationInfo(BaseModel):
    """
    Model for validation results.
    """
    is_valid: bool = Field(..., description="Whether the extracted data passed validation")
    warnings: List[str] = Field(default_factory=list, description="Non-critical validation warnings")
    errors: List[str] = Field(default_factory=list, description="Critical validation errors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "warnings": ["License expired on 25-12-2024"],
                "errors": []
            }
        }


# ============================================================================
# OCR RESPONSE MODEL
# ============================================================================

class OCRResponse(BaseModel):
    """
    Complete OCR API response.
    This is what the endpoint returns to the client.
    """
    success: bool = Field(..., description="Whether OCR processing was successful")
    message: str = Field(..., description="Human-readable status message")
    data: Optional[TradeLicenseData] = Field(None, description="Extracted license data")
    validation: Optional[ValidationInfo] = Field(None, description="Validation results")
    processing_time: Optional[float] = Field(None, description="Time taken to process (seconds)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Trade license processed successfully",
                "data": {
                    "license_number": "CN-2883802",
                    "company_name": "TIGER FIRE FOR SECURITY SYSTEM",
                    "establishment_date": "14-10-2019",
                    "expiry_date": "25-12-2024",
                    "manager_name": "NAJLA ANTAR ABDULLA ALJABERI",
                    "owners": [
                        {
                            "name": "NAJLA ANTAR ABDULLA ALJABERI",
                            "share_percentage": "100%"
                        }
                    ]
                },
                "validation": {
                    "is_valid": True,
                    "warnings": [],
                    "errors": []
                },
                "processing_time": 2.34,
                "timestamp": "2026-01-09T10:30:00.123456"
            }
        }


# ============================================================================
# ERROR RESPONSE MODEL
# ============================================================================

class ErrorResponse(BaseModel):
    """
    Error response model for failed requests.
    """
    success: bool = Field(False, description="Always false for errors")
    message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Failed to process image",
                "error_type": "ProcessingError",
                "details": {
                    "reason": "No detections found in image"
                },
                "timestamp": "2026-01-09T10:30:00.123456"
            }
        }