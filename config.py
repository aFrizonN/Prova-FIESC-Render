"""
Configuration settings for the Prescriptive Maintenance System.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROVA_DIR = BASE_DIR / "Prova"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "db"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, DOCS_DIR, DB_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data paths
RAW_CSV_PATH = PROVA_DIR / "banner.csv"
PROCESSED_CSV_PATH = DATA_DIR / "banner_processed.parquet"
DB_PATH = DB_DIR / "maintenance.db"

# Model paths
CLASSIFIER_PATH = MODELS_DIR / "fault_classifier.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.0-flash"

# RAG & Chroma settings
CHROMA_COLLECTION_NAME = "maintenance_technical_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# App settings
APP_PORT = int(os.getenv("PORT", 8501))
API_PORT = int(os.getenv("API_PORT", 8000))
SIMILARITY_TOP_K = 10
