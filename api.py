"""
FastAPI REST API for Industrial Prescriptive Maintenance System.
Provides endpoints for real-time sensor event ingestion, ML classification,
similarity search, RAG prescriptive reasoning, and documentation management.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging

import config
from src.classifier import get_classifier
from src.similarity import get_similarity_engine
from src.rag_engine import get_rag_engine
from src.document_processor import get_document_processor
from src.database import get_kpis, get_fault_distribution, get_time_series_faults, log_chat_message

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Industrial Prescriptive Maintenance API",
    version="1.0.0",
    description="Full-stack AI API for predictive fault classification, historical similarity search, and prescriptive RAG diagnostics.",
)

# Enable CORS for cross-origin industrial dashboards and frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class SensorEventInput(BaseModel):
    id: Optional[int] = Field(None, example=114387)
    created_at: Optional[str] = Field(None, example="2026-06-01 21:32:53.911176+00:00")
    rpm: float = Field(..., example=1000.0)
    temperature_c: float = Field(..., example=24.69)
    temperature_f: Optional[float] = Field(76.44, example=76.44)
    z_rms_velocity_mm_s: float = Field(..., example=1.517)
    x_rms_velocity_mm_s: float = Field(..., example=2.0)
    z_rms_velocity_in_s: Optional[float] = Field(0.0597)
    x_rms_velocity_in_s: Optional[float] = Field(0.0787)
    z_peak_acceleration_g: float = Field(..., example=0.484)
    x_peak_acceleration_g: float = Field(..., example=0.631)
    z_peak_vel_comp_freq_hz: Optional[float] = Field(61.0)
    x_peak_vel_comp_freq_hz: Optional[float] = Field(61.0)
    z_rms_acceleration_g: float = Field(..., example=0.09)
    x_rms_acceleration_g: float = Field(..., example=0.114)
    z_kurtosis: float = Field(..., example=2.392)
    x_kurtosis: float = Field(..., example=2.77)
    z_crest_factor: float = Field(..., example=3.747)
    x_crest_factor: float = Field(..., example=4.269)
    z_peak_velocity_mm_s: Optional[float] = Field(2.146)
    x_peak_velocity_mm_s: Optional[float] = Field(2.829)
    z_peak_velocity_in_s: Optional[float] = Field(0.0844)
    x_peak_velocity_in_s: Optional[float] = Field(0.1113)
    z_high_freq_rms_accel_g: float = Field(..., example=0.129)
    x_high_freq_rms_accel_g: float = Field(..., example=0.147)
    fault: Optional[str] = Field(None, example="cocked_rotor_2")


class PredictionResponse(BaseModel):
    predicted_category: str
    confidence: float
    probabilities: Dict[str, float]


class PrescriptiveResponse(BaseModel):
    predicted_category: str
    confidence: float
    has_document: bool
    is_operational_normal: bool
    prescription: str
    referenced_docs: List[str]
    similarity_summary: Dict[str, Any]


class ChatMessageInput(BaseModel):
    session_id: str = Field(..., example="session_123")
    message: str = Field(..., example="Quais ferramentas preciso para alinhar o motor?")
    event_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# Endpoints
@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint confirming API status and ML subsystem availability."""
    classifier = get_classifier()
    return {
        "status": "healthy",
        "service": "Prescriptive Maintenance AI",
        "model_loaded": classifier.model is not None,
        "classes_count": len(classifier.classes) if classifier.classes else 0,
    }


@app.get("/api/v1/kpis", tags=["Dashboard"])
def get_dashboard_kpis():
    """Returns top-level KPIs for real-time industrial supervision."""
    return get_kpis()


@app.get("/api/v1/charts/distribution", tags=["Dashboard"])
def get_fault_distribution_chart():
    """Returns fault categories distribution data."""
    df = get_fault_distribution()
    return df.to_dict(orient="records")


@app.get("/api/v1/charts/timeline", tags=["Dashboard"])
def get_fault_timeline_chart():
    """Returns time-series evolution of detected anomalies."""
    df = get_time_series_faults()
    return df.to_dict(orient="records")


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_fault(event: SensorEventInput):
    """Predicts the macro fault category and confidence from vibration telemetry."""
    classifier = get_classifier()
    try:
        category, conf, probas = classifier.predict(event.model_dump())
        return PredictionResponse(
            predicted_category=category,
            confidence=conf,
            probabilities=probas,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/similarity", tags=["Inference"])
def find_similar_events(event: SensorEventInput, top_k: int = 10):
    """Finds top-K historical events with similar vibration/spectral profiles."""
    similarity_engine = get_similarity_engine()
    try:
        results = similarity_engine.find_similar(event.model_dump(), top_k=top_k)
        return results
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/prescribe", response_model=PrescriptiveResponse, tags=["Prescription"])
def full_prescriptive_pipeline(event: SensorEventInput):
    """
    Executes the end-to-end prescriptive maintenance pipeline:
    1. Feature Engineering & Scaling
    2. ML Fault Classification
    3. Historical Similarity Search & Context Extraction
    4. Guardrailed Technical Documentation RAG Reasoning
    """
    classifier = get_classifier()
    similarity_engine = get_similarity_engine()
    rag_engine = get_rag_engine()

    try:
        event_dict = event.model_dump()
        category, conf, _ = classifier.predict(event_dict)
        sim_res = similarity_engine.find_similar(event_dict)
        presc_res = rag_engine.generate_prescription(
            event_data=event_dict,
            predicted_category=category,
            confidence=conf,
            similarity_results=sim_res,
        )

        return PrescriptiveResponse(
            predicted_category=category,
            confidence=conf,
            has_document=presc_res["has_document"],
            is_operational_normal=presc_res["is_operational_normal"],
            prescription=presc_res["prescription"],
            referenced_docs=presc_res.get("referenced_docs", []),
            similarity_summary=sim_res,
        )
    except Exception as e:
        logger.error(f"Prescriptive pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(chat_input: ChatMessageInput):
    """Interactive conversational chat endpoint with strict technical domain guardrails."""
    rag_engine = get_rag_engine()
    try:
        # Build simple turn history
        history = [{"role": "user", "content": chat_input.message}]
        response_text = rag_engine.chat_with_technician(
            user_message=chat_input.message,
            chat_history=history,
            current_event_context=chat_input.event_context,
        )
        log_chat_message(
            session_id=chat_input.session_id,
            role="user",
            message=chat_input.message,
            event_context=chat_input.event_context,
        )
        log_chat_message(
            session_id=chat_input.session_id,
            role="assistant",
            message=response_text,
            event_context=chat_input.event_context,
        )
        return ChatResponse(
            response=response_text,
            session_id=chat_input.session_id,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/upload", tags=["Documentation"])
async def upload_document(
    file: UploadFile = File(...),
    fault_category: str = Form(...),
    title: str = Form(...),
):
    """Uploads and indexes a new technical manual PDF to expand the RAG knowledge base."""
    dp = get_document_processor()
    try:
        content = await file.read()
        res = dp.add_new_document(
            uploaded_file_bytes=content,
            filename=file.filename,
            fault_category=fault_category,
            custom_title=title,
        )
        return res
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=config.API_PORT, reload=False)
