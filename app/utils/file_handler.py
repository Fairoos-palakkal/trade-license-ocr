"""
File Handler - Utilities for file upload and management
"""

import os
import uuid
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile
import shutil


# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file.
    
    Checks:
    1. File has a name
    2. File extension is allowed
    3. File is not empty
    
    Args:
        file: Uploaded file from FastAPI
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        valid, error = validate_file(uploaded_file)
        if not valid:
            raise ValueError(error)
    """
    # Check if file exists
    if not file:
        return False, "No file provided"
    
    # Check if file has a name
    if not file.filename:
        return False, "File has no name"
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"File type {file_ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, None


# ============================================================================
# FILE SAVING
# ============================================================================

def save_upload_file(file: UploadFile) -> str:
    """
    Save uploaded file to disk with unique filename.
    
    Steps:
    1. Validate file
    2. Generate unique filename (UUID)
    3. Save to uploads directory
    4. Return file path
    
    Args:
        file: Uploaded file from FastAPI
        
    Returns:
        Path to saved file
        
    Raises:
        ValueError: If file validation fails
        
    Example:
        file_path = save_upload_file(uploaded_file)
        # Returns: "uploads/abc123-def456.jpg"
    """
    # Validate file
    is_valid, error = validate_file(file)
    if not is_valid:
        raise ValueError(error)
    
    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_ext = Path(file.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise IOError(f"Failed to save file: {str(e)}")
    finally:
        file.file.close()
    
    return str(file_path)


# ============================================================================
# FILE CLEANUP
# ============================================================================

def cleanup_file(file_path: str) -> None:
    """
    Delete a file from disk.
    
    Used to clean up temporary uploaded files after processing.
    
    Args:
        file_path: Path to file to delete
        
    Example:
        cleanup_file("uploads/abc123.jpg")
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️  Cleaned up: {file_path}")
    except Exception as e:
        print(f"⚠️  Failed to cleanup {file_path}: {str(e)}")


# ============================================================================
# FILE SIZE CHECK
# ============================================================================

def check_file_size(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> Tuple[bool, Optional[str]]:
    """
    Check if file size is within allowed limit.
    
    Args:
        file: Uploaded file
        max_size: Maximum allowed size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Read a bit to check size (don't load entire file)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()  # Get position (file size)
    file.file.seek(0)  # Reset to beginning
    
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        return False, f"File too large: {actual_mb:.2f}MB (max: {max_mb:.2f}MB)"
    
    return True, None