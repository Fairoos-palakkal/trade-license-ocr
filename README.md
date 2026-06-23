<p align="center">
  <h1 align="center">📜 Trade License OCR</h1>
  <p align="center">
    <b>AI-Powered Structured Data Extraction from Abu Dhabi Trade Licenses</b>
  </p>
  <p align="center">
    <a href="#-features"><img src="https://img.shields.io/badge/Detection-YOLOv8-FF6F00?style=for-the-badge&logo=yolo" alt="YOLOv8"></a>
    <a href="#-features"><img src="https://img.shields.io/badge/OCR-Tesseract-4285F4?style=for-the-badge" alt="OCR"></a>
    <a href="#-tech-stack"><img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
    <a href="#-features"><img src="https://img.shields.io/badge/Model-Roboflow-6706CE?style=for-the-badge&logo=roboflow" alt="Roboflow"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/language-Python%20100%25-blue" alt="Python 100%">
    <img src="https://img.shields.io/badge/framework-FastAPI-009688" alt="FastAPI">
  </p>
</p>

---

An intelligent document processing system that automatically extracts structured data from **Abu Dhabi Trade License** images and PDFs. Upload a trade license document, and the system uses a **YOLOv8 object detection** model (trained via Roboflow) to locate key fields, **OCR** to extract text from each detected region, and an **intelligent parser** to clean, validate, and structure the extracted data into a standardized JSON format.

> **Use Cases:** Government document digitization · Trade license verification · Business registration automation · KYC compliance · Document archival systems

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚡ Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run the Server](#run-the-server)
- [🔌 API Reference](#-api-reference)
  - [Process Trade License](#post-ocrtrade-license)
  - [Sample Response](#sample-response)
- [🧪 Testing](#-testing)
- [📊 Pipeline Output](#-pipeline-output)
- [🖥️ Frontend](#️-frontend)
- [⚙️ Configuration](#️-configuration)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

| Category | Feature | Details |
|----------|---------|---------|
| 🎯 **Detection** | YOLOv8 Object Detection | Custom-trained Roboflow model to locate license fields (license number, company name, dates, owners, etc.) |
| 🔤 **OCR** | Text Extraction | Per-field OCR from cropped detection regions with confidence scoring |
| 🧹 **Parsing** | Intelligent Post-Processing | Field mapping, text cleaning, date normalization, ownership structuring |
| 📋 **Validation** | Data Quality Checks | Confidence thresholds, format validation, completeness scoring |
| 🌐 **API** | FastAPI REST Endpoint | Single-endpoint file upload → structured JSON response |
| 📄 **Formats** | Multi-Format Support | JPG, JPEG, PNG, and PDF input support |
| 🏷️ **Schema** | Pydantic v2 Models | Strongly-typed request/response schemas with validation |
| 🐞 **Debug** | Crop Saving | Save detected field crops for debugging and quality inspection |
| 🔗 **CORS** | Cross-Origin Support | Frontend-ready with full CORS middleware |
| 📊 **Metrics** | Processing Telemetry | Extraction time, confidence scores, field detection counts |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Client (Frontend / cURL)                      │
│               Upload trade license image / PDF                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │  POST /ocr/trade-license
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (app/)                          │
│                                                                   │
│  ┌─────────────┐                                                  │
│  │  routes.py  │  ← API endpoint, file validation, orchestration  │
│  └──────┬──────┘                                                  │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              Processing Pipeline (services/)                 │ │
│  │                                                              │ │
│  │  ┌────────────────┐   ┌────────────────┐   ┌──────────────┐ │ │
│  │  │  ocr_service   │──▶│  ocr_pipeline  │──▶│   parser     │ │ │
│  │  │                │   │                │   │   service     │ │ │
│  │  │ • Load YOLO    │   │ • Crop fields  │   │ • Field map  │ │ │
│  │  │ • Run detect   │   │ • Run OCR      │   │ • Clean text │ │ │
│  │  │ • Get bboxes   │   │ • Get text     │   │ • Normalize  │ │ │
│  │  │ • Confidence   │   │ • Confidence   │   │ • Structure  │ │ │
│  │  └────────────────┘   └────────────────┘   └──────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────────┐  ┌──────────────────┐                       │
│  │  schemas.py      │  │  file_handler.py │                       │
│  │  Pydantic v2     │  │  Upload / Save   │                       │
│  │  Response Models │  │  Cleanup / Size  │                       │
│  └──────────────────┘  └──────────────────┘                       │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                   YOLOv8 Model (ml_models/)                       │
│               Roboflow-trained · model.pt (~22MB)                 │
│         Custom classes: license fields on Abu Dhabi docs          │
└──────────────────────────────────────────────────────────────────┘
```

**Pipeline Flow:**

1. **Upload** → Client sends trade license image/PDF via `POST /ocr/trade-license`
2. **Validate** → File type check (JPG/PNG/PDF), size check (≤10MB), save to `uploads/`
3. **Detect** → YOLOv8 model runs inference to detect field regions (bounding boxes)
4. **Crop & OCR** → Each detected region is cropped and fed through OCR for text extraction
5. **Parse** → Raw OCR text is cleaned, mapped to fields, dates normalized, owners structured
6. **Respond** → Structured JSON with extracted fields, confidence scores, and validation info

---

## 📁 Project Structure

```
trade-license-ocr/
├── app/                              # Main application package
│   ├── __init__.py
│   ├── main.py                       # FastAPI app factory, CORS, router registration
│   │
│   ├── api/                          # API layer
│   │   ├── __init__.py
│   │   └── routes.py                 # POST /ocr/trade-license endpoint
│   │
│   ├── models/                       # Data models
│   │   ├── __init__.py
│   │   └── schemas.py                # Pydantic v2: Owner, TradeLicenseData, OCRResponse
│   │
│   ├── services/                     # Business logic
│   │   ├── __init__.py
│   │   ├── ocr_service.py            # YOLO model loading & inference
│   │   ├── ocr_pipeline.py           # Detection → Crop → OCR text extraction
│   │   └── parser_service.py         # Text cleaning, field mapping, structuring
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       └── file_handler.py           # File upload, save, size check, cleanup
│
├── ml_models/                        # Machine learning models
│   ├── README.md
│   └── roboflow_model/
│       └── model.pt                  # YOLOv8 weights (~22MB, trained via Roboflow)
│
├── uploads/                          # Runtime upload directory
│   ├── debug_crops/                  # Saved field crop images for debugging
│   ├── dummy_license.jpg             # Test image for development
│   ├── trade_license_sample.jpg      # Sample trade license image
│   ├── ocr_results.json              # Sample raw OCR output
│   └── final_output.json             # Sample structured output
│
├── frontend/                         # Frontend application (submodule)
│
├── run.py                            # Server entry point (uvicorn launcher)
├── test_model.py                     # Model loading & inference test
├── test_ocr_pipeline.py              # End-to-end OCR pipeline test
├── test_parser.py                    # Parser service test (YOLO → OCR → Parse)
└── requirements.txt                  # Pinned Python dependencies
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Object Detection** | [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics) | Locate field regions on trade license documents |
| **Model Training** | [Roboflow](https://roboflow.com/) | Annotate, train, and export custom YOLO model |
| **OCR Engine** | Text Extraction | Read text from detected/cropped regions |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) 0.104 | Async REST API with auto-generated docs |
| **Server** | [Uvicorn](https://www.uvicorn.org/) | High-performance ASGI server |
| **Schemas** | [Pydantic v2](https://docs.pydantic.dev/) | Request/response validation and serialization |
| **Deep Learning** | [PyTorch](https://pytorch.org/) 2.1 | Model inference runtime |
| **Image Processing** | [Pillow](https://pillow.readthedocs.io/) + NumPy | Image manipulation and preprocessing |
| **Language** | Python 3.10+ | 100% Python codebase |

---

## ⚡ Quick Start

### Prerequisites

- **Python** 3.10+
- **pip** for package management
- **Git** for cloning

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Fairoos-palakkal/trade-license-ocr.git
cd trade-license-ocr

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The YOLOv8 model weights (`ml_models/roboflow_model/model.pt`) are included in the repository (~22MB). No separate download needed.

### Run the Server

```bash
# Option 1: Using run.py (recommended)
python run.py

# Option 2: Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔌 API Reference

### `POST /ocr/trade-license`

Upload a trade license image or PDF to extract structured data.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | `UploadFile` | ✅ | Trade license image (JPG, JPEG, PNG) or PDF |

**Constraints:**
- Maximum file size: **10MB**
- Supported formats: `JPG`, `JPEG`, `PNG`, `PDF`

#### cURL Example

```bash
curl -X POST "http://localhost:8000/ocr/trade-license" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/trade_license.jpg"
```

#### Python Example

```python
import requests

url = "http://localhost:8000/ocr/trade-license"
files = {"file": open("trade_license.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Sample Response

```json
{
  "success": true,
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
    "fields_extracted": 5,
    "confidence_avg": 0.85,
    "processing_time_ms": 1230
  }
}
```

### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether extraction was successful |
| `data.license_number` | `string \| null` | Trade license number (e.g., `CN-2883802`) |
| `data.company_name` | `string \| null` | Registered company/trade name |
| `data.establishment_date` | `string \| null` | Date of establishment |
| `data.expiry_date` | `string \| null` | License expiry date |
| `data.manager_name` | `string \| null` | Name of the manager |
| `data.owners` | `Owner[]` | List of owners with name and share percentage |
| `validation.fields_extracted` | `int` | Number of fields successfully extracted |
| `validation.confidence_avg` | `float` | Average detection confidence score |
| `validation.processing_time_ms` | `int` | Total processing time in milliseconds |

---

## 📊 Pipeline Output

### Raw OCR Results (per-field)

Each detected field produces a result with detection and OCR confidence:

```json
{
  "EXPIRY_DATE": {
    "text": "13/01/2024",
    "detection_confidence": 0.708,
    "ocr_confidence": 0.999,
    "bbox": { "x1": 585, "y1": 616, "x2": 747, "y2": 655 }
  },
  "COMPANY_NAME": {
    "text": "AL RAKHA CONTRACTING & GENERAL TRANSPORT LLC",
    "detection_confidence": 0.623,
    "ocr_confidence": 0.95,
    "bbox": { "x1": 120, "y1": 280, "x2": 890, "y2": 330 }
  }
}
```

### Detected Field Classes

The YOLOv8 model is trained to detect the following field regions on Abu Dhabi trade licenses:

| Class | Description | Example Value |
|-------|-------------|---------------|
| `LICENSE_NUMBER` | Trade license number | `CN-2883802` |
| `COMPANY_NAME` | Registered company name | `TIGER FIRE FOR SECURITY SYSTEM` |
| `ESTABLISHMENT_DATE` | Date of establishment | `14-10-2019` |
| `EXPIRY_DATE` | License expiry date | `25-12-2024` |
| `MANAGER_NAME` | Manager's full name | `NAJLA ANTAR ABDULLA ALJABERI` |
| `OWNER_NAME` | Owner name(s) | `NAJLA ANTAR ABDULLA ALJABERI` |
| `SHARE_PERCENTAGE` | Ownership share | `100%` |

---

## 🧪 Testing

The project includes three standalone test scripts for verifying each stage of the pipeline:

```bash
# Test 1: Model loading and inference
python test_model.py

# Test 2: Complete OCR pipeline (YOLO → Crop → OCR)
python test_ocr_pipeline.py

# Test 3: Full pipeline with parser (YOLO → OCR → Parse → Structure)
python test_parser.py
```

| Test Script | What it Tests | Pipeline Stage |
|-------------|--------------|----------------|
| `test_model.py` | Model loading, dummy image inference | Detection only |
| `test_ocr_pipeline.py` | YOLO detection → field cropping → OCR extraction | Detection + OCR |
| `test_parser.py` | Complete pipeline → cleaning → structuring → JSON output | End-to-end |

> **Tip:** Sample images are included in `uploads/` for testing. Replace `trade_license_sample.jpg` with your own Abu Dhabi trade license image for real-world testing.

---

## 🖥️ Frontend

The project includes a frontend application (referenced as a submodule) that provides a web interface for:

- 📤 Uploading trade license images via drag-and-drop
- 📊 Viewing extracted structured data in a clean format
- 🔍 Inspecting detection bounding boxes and confidence scores
- 📋 Copying/exporting results as JSON

> The frontend connects to the FastAPI backend via the CORS-enabled REST API.

---

## ⚙️ Configuration

### API Configuration (`app/main.py`)

| Setting | Value | Description |
|---------|-------|-------------|
| `title` | `Trade License OCR API` | API title in Swagger docs |
| `version` | `1.0.0` | API version |
| `docs_url` | `/docs` | Swagger UI path |
| `redoc_url` | `/redoc` | ReDoc documentation path |
| CORS | `allow_origins=["*"]` | Allow all origins (restrict in production) |

### Server Configuration (`run.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `0.0.0.0` | Listen on all interfaces |
| `port` | `8000` | Server port |
| `reload` | `True` | Auto-reload on code changes (dev mode) |

### Model Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| Model path | `ml_models/roboflow_model/model.pt` | YOLOv8 weights location |
| Model size | ~22MB | Pre-trained PyTorch weights |
| Confidence threshold | `0.25` | Default detection confidence |
| Max file size | `10MB` | Upload size limit |

---

## 🗺️ Roadmap

- [x] YOLOv8 field detection with Roboflow model
- [x] OCR pipeline with cropping and text extraction
- [x] Intelligent text parsing and field structuring
- [x] FastAPI REST endpoint with Pydantic schemas
- [x] File upload handling with validation
- [x] Debug crop saving for inspection
- [x] Test scripts for each pipeline stage
- [ ] PDF multi-page support
- [ ] Arabic text extraction (bilingual license support)
- [ ] Batch processing endpoint for multiple documents
- [ ] Database integration for processed records
- [ ] Confidence-based manual review flagging
- [ ] Docker containerization
- [ ] Frontend dashboard with result visualization

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** your feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code style
- Add test scripts for new pipeline components
- Use Pydantic models for all API schemas
- Use type hints throughout the codebase
- Keep ML model files tracked with Git LFS for large models

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/Fairoos-palakkal">Fairoos Palakkal</a></b>
  <br>
  <sub>If you found this project useful, please consider giving it a ⭐</sub>
</p>
