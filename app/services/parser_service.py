"""
Parser Service - Post-processing and field mapping
Converts raw OCR text into structured, clean data
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime


# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean OCR text by removing noise and normalizing whitespace.
    
    Steps:
    1. Remove null/None
    2. Strip leading/trailing whitespace
    3. Normalize internal whitespace (multiple spaces → single space)
    4. Remove common OCR artifacts
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
        
    Examples:
        " CN - 2883802  " → "CN-2883802"
        "TIGER  FIRE   FOR" → "TIGER FIRE FOR"
    """
    if not text or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Strip whitespace
    text = text.strip()
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common OCR artifacts
    # These are characters that OCR sometimes hallucinates
    artifacts = ['|', '~', '`', '�']
    for artifact in artifacts:
        text = text.replace(artifact, '')
    
    return text


def clean_license_number(text: str) -> str:
    """
    Clean and normalize license number.
    
    Abu Dhabi license numbers are typically: CN-XXXXXXX
    
    Args:
        text: Raw license number text
        
    Returns:
        Cleaned license number
        
    Examples:
        " CN - 2883802 " → "CN-2883802"
        "CN 2883802" → "CN-2883802"
    """
    text = clean_text(text)
    
    # Remove all spaces
    text = text.replace(' ', '')
    
    # Ensure there's a dash between letters and numbers
    # Pattern: 2+ letters, optional dash, digits
    text = re.sub(r'([A-Z]{2,})[-\s]*(\d+)', r'\1-\2', text, flags=re.IGNORECASE)
    
    # Convert to uppercase
    text = text.upper()
    
    return text


def clean_company_name(text: str) -> str:
    """
    Clean company/trade name.
    
    Args:
        text: Raw company name
        
    Returns:
        Cleaned company name
        
    Examples:
        "TIGER  FIRE FOR  SECURITY SYSTEM" → "TIGER FIRE FOR SECURITY SYSTEM"
    """
    text = clean_text(text)
    
    # Capitalize properly (some OCR returns all caps or mixed case)
    # Keep as-is if already capitalized, otherwise title case
    # For Abu Dhabi licenses, names are usually ALL CAPS, so keep that
    text = text.upper()
    
    return text


def clean_person_name(text: str) -> str:
    """
    Clean person names (owner, manager).
    
    Args:
        text: Raw person name
        
    Returns:
        Cleaned name
        
    Examples:
        "NAJLA  ANTAR ABDULLA  ALJABERI" → "NAJLA ANTAR ABDULLA ALJABERI"
    """
    text = clean_text(text)
    
    # Remove any percentages that might be attached to name
    text = re.sub(r'\d+\s*%.*$', '', text)
    
    # Remove common suffixes that might be OCR errors
    text = re.sub(r'\s+(Owner|Manager|Partner)$', '', text, flags=re.IGNORECASE)
    
    text = text.strip().upper()
    
    return text


# ============================================================================
# DATE PARSING AND NORMALIZATION
# ============================================================================

def parse_date(date_text: str) -> Optional[str]:
    """
    Parse and normalize date to DD-MM-YYYY format.
    
    Handles various formats:
    - DD/MM/YYYY
    - DD-MM-YYYY
    - DD.MM.YYYY
    - DD MM YYYY
    
    Args:
        date_text: Raw date string
        
    Returns:
        Normalized date as "DD-MM-YYYY" or None if invalid
        
    Examples:
        "25/12/2024" → "25-12-2024"
        "09-01-2018" → "09-01-2018"
        "14.10.2019" → "14-10-2019"
    """
    if not date_text:
        return None
    
    # Clean text
    date_text = clean_text(date_text)
    
    # Remove any non-date characters
    date_text = re.sub(r'[^\d/\-\.\s]', '', date_text)
    
    # Try different date formats
    formats = [
        r'(\d{2})[/\-\.\s](\d{2})[/\-\.\s](\d{4})',  # DD/MM/YYYY
        r'(\d{1})[/\-\.\s](\d{2})[/\-\.\s](\d{4})',   # D/MM/YYYY
        r'(\d{2})[/\-\.\s](\d{1})[/\-\.\s](\d{4})',   # DD/M/YYYY
    ]
    
    for pattern in formats:
        match = re.search(pattern, date_text)
        if match:
            day = match.group(1).zfill(2)  # Add leading zero if needed
            month = match.group(2).zfill(2)
            year = match.group(3)
            
            # Validate date
            try:
                # Check if date is valid
                datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y")
                return f"{day}-{month}-{year}"
            except ValueError:
                # Invalid date, continue to next format
                continue
    
    # If no format matched, return None
    return None


# ============================================================================
# OWNER PARSING
# ============================================================================

def extract_percentage(text: str) -> Optional[str]:
    """
    Extract percentage from text.
    
    Args:
        text: Text that might contain a percentage
        
    Returns:
        Percentage string (e.g., "100%") or None
        
    Examples:
        "NAJLA ANTAR 100%" → "100%"
        "50 %" → "50%"
        "Owner 25.5%" → "25.5%"
    """
    if not text:
        return None
    
    # Look for number followed by % (with optional space)
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if match:
        percentage = match.group(1)
        return f"{percentage}%"
    
    return None


def parse_owner_text(text: str) -> Dict[str, str]:
    """
    Parse owner text that contains both name and percentage.
    
    Args:
        text: Raw owner text (e.g., "NAJLA ANTAR 100%")
        
    Returns:
        Dictionary with 'name' and 'share_percentage'
        
    Examples:
        "NAJLA ANTAR ABDULLA ALJABERI 100%" →
        {'name': 'NAJLA ANTAR ABDULLA ALJABERI', 'share_percentage': '100%'}
    """
    # Extract percentage
    percentage = extract_percentage(text)
    
    # Remove percentage from text to get name
    name_text = re.sub(r'\d+(?:\.\d+)?\s*%', '', text)
    name = clean_person_name(name_text)
    
    return {
        'name': name,
        'share_percentage': percentage or '100%'  # Default to 100% if not found
    }


def consolidate_owners(ocr_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract and consolidate all owner information.
    
    Handles multiple owners from different OCR detections:
    - owner, owner_1, owner_2, etc.
    - partner, partner_1, partner_2, etc.
    
    Args:
        ocr_results: Raw OCR results dictionary
        
    Returns:
        List of owner dictionaries
        
    Examples:
        Input:
        {
            'owner_1': {'text': 'NAJLA ANTAR 50%', ...},
            'owner_2': {'text': 'AHMED HASSAN 50%', ...}
        }
        
        Output:
        [
            {'name': 'NAJLA ANTAR', 'share_percentage': '50%'},
            {'name': 'AHMED HASSAN', 'share_percentage': '50%'}
        ]
    """
    owners = []
    
    # Look for all owner-related fields
    owner_keys = [
        'owner', 'owner_1', 'owner_2', 'owner_3', 'owner_4',
        'partner', 'partner_1', 'partner_2', 'partner_3',
        'owners', 'partners'
    ]
    
    for key in owner_keys:
        if key in ocr_results:
            text = ocr_results[key].get('text', '')
            if text and text.strip():
                owner_data = parse_owner_text(text)
                if owner_data['name']:  # Only add if name is not empty
                    owners.append(owner_data)
    
    # If no owners found, return empty list (not an error - might be single proprietor)
    return owners


# ============================================================================
# FIELD MAPPING
# ============================================================================

def map_fields(ocr_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Map raw OCR results to structured output format.
    
    This is the main function that orchestrates all cleaning and mapping.
    
    Args:
        ocr_results: Raw OCR results from ocr_pipeline.process_detections()
                    Format: {field_name: {'text': '...', 'confidence': 0.9, ...}}
        
    Returns:
        Structured dictionary with clean, normalized data
        
    Output format:
        {
            'license_number': 'CN-2883802',
            'company_name': 'TIGER FIRE FOR SECURITY SYSTEM',
            'establishment_date': '14-10-2019',
            'expiry_date': '25-12-2024',
            'manager_name': 'NAJLA ANTAR ABDULLA ALJABERI',
            'owners': [
                {
                    'name': 'NAJLA ANTAR ABDULLA ALJABERI',
                    'share_percentage': '100%'
                }
            ]
        }
    """
    print("\n" + "=" * 60)
    print("🗂️  MAPPING FIELDS TO STRUCTURED FORMAT")
    print("=" * 60)
    
    # Initialize output structure
    output = {
        'license_number': None,
        'company_name': None,
        'establishment_date': None,
        'expiry_date': None,
        'manager_name': None,
        'owners': []
    }
    
    # Helper function to get text from OCR results
    def get_field_text(field_names):
        for key, value in ocr_results.items():
            if key.lower() in field_names:
                text = value.get('text', '')
                if text and text.strip():
                    return text
        return None
    
    # MAP: License Number
    license_text = get_field_text(['license_number', 'license_no', 'licence_number'])
    if license_text:
        output['license_number'] = clean_license_number(license_text)
        print(f"   ✅ License Number: {output['license_number']}")
    else:
        print(f"   ⚠️  License Number: Not found")
    
    # MAP: Company Name / Trade Name
    company_text = get_field_text([
        'company_name', 'trade_name', 'company', 'trade', 
        'establishment_name', 'business_name'
    ])
    if company_text:
        output['company_name'] = clean_company_name(company_text)
        print(f"   ✅ Company Name: {output['company_name']}")
    else:
        print(f"   ⚠️  Company Name: Not found")
    
    # MAP: Establishment Date
    est_date_text = get_field_text([
        'establishment_date', 'est_date', 'founded_date', 
        'establishment', 'date_established'
    ])
    if est_date_text:
        output['establishment_date'] = parse_date(est_date_text)
        print(f"   ✅ Establishment Date: {output['establishment_date']}")
    else:
        print(f"   ⚠️  Establishment Date: Not found")
    
    # MAP: Expiry Date
    expiry_text = get_field_text([
        'expiry_date', 'expiration_date', 'expire_date', 
        'valid_until', 'expiry'
    ])
    if expiry_text:
        output['expiry_date'] = parse_date(expiry_text)
        print(f"   ✅ Expiry Date: {output['expiry_date']}")
    else:
        print(f"   ⚠️  Expiry Date: Not found")
    
    # MAP: Manager Name
    manager_text = get_field_text([
        'manager', 'manager_name', 'authorized_signatory',
        'signatory', 'responsible_person'
    ])
    if manager_text:
        output['manager_name'] = clean_person_name(manager_text)
        print(f"   ✅ Manager Name: {output['manager_name']}")
    else:
        print(f"   ⚠️  Manager Name: Not found")
    
    # MAP: Owners
    owners = consolidate_owners(ocr_results)
    if owners:
        output['owners'] = owners
        print(f"   ✅ Owners: {len(owners)} found")
        for i, owner in enumerate(owners, 1):
            print(f"      {i}. {owner['name']} ({owner['share_percentage']})")
    else:
        print(f"   ⚠️  Owners: Not found")
    
    print("=" * 60 + "\n")
    
    return output


# ============================================================================
# VALIDATION
# ============================================================================

def validate_parsed_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parsed data and add validation status.
    
    Checks:
    - License number format
    - Date validity
    - Owner shares sum to 100% (if numeric percentages)
    
    Args:
        data: Parsed data dictionary
        
    Returns:
        Same dictionary with added 'validation' field
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check license number format
    if data['license_number']:
        if not re.match(r'^[A-Z]{2}-\d+$', data['license_number']):
            validation['warnings'].append(
                f"License number format unusual: {data['license_number']}"
            )
    else:
        validation['errors'].append("License number is missing")
        validation['is_valid'] = False
    
    # Check if license is expired
    if data['expiry_date']:
        try:
            expiry = datetime.strptime(data['expiry_date'], "%d-%m-%Y")
            if expiry < datetime.now():
                validation['warnings'].append(
                    f"License expired on {data['expiry_date']}"
                )
        except:
            pass
    
    # Check owner shares (if numeric)
    if data['owners']:
        try:
            total_shares = 0
            for owner in data['owners']:
                share_str = owner['share_percentage'].replace('%', '')
                total_shares += float(share_str)
            
            if abs(total_shares - 100.0) > 0.1:  # Allow tiny floating point error
                validation['warnings'].append(
                    f"Owner shares total {total_shares}% (expected 100%)"
                )
        except:
            # If we can't parse percentages, skip validation
            pass
    
    data['validation'] = validation
    
    return data


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_and_structure(ocr_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function: Process raw OCR results into structured, validated output.
    
    This combines all steps:
    1. Map fields
    2. Clean and normalize
    3. Validate
    
    Args:
        ocr_results: Raw OCR results from ocr_pipeline.process_detections()
        
    Returns:
        Complete structured output with validation
    """
    print("\n" + "=" * 60)
    print("🔄 POST-PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Map and clean fields
    structured_data = map_fields(ocr_results)
    
    # Step 2: Validate
    print("✔️  Validating parsed data...")
    validated_data = validate_parsed_data(structured_data)
    
    # Print validation results
    validation = validated_data['validation']
    if validation['is_valid']:
        print("   ✅ Data is valid")
    else:
        print("   ❌ Data has errors:")
        for error in validation['errors']:
            print(f"      - {error}")
    
    if validation['warnings']:
        print("   ⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"      - {warning}")
    
    print("=" * 60 + "\n")
    
    return validated_data