"""
End-to-end training, database ingestion, document processing, and vector indexing pipeline.
"""

import time
import json
import logging
import sys

from src.database import init_db, ingest_dataframe_to_db
from src.data_processing import load_and_preprocess_raw_data
from src.classifier import FaultClassifier
from src.similarity import SimilarityEngine
from src.document_processor import DocumentProcessor
from src.rag_engine import get_rag_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("TrainPipeline")


def run_full_pipeline():
    start_total = time.time()
    logger.info("=" * 70)
    logger.info("STARTING PRESCRIPTIVE MAINTENANCE AI PIPELINE INITIALIZATION")
    logger.info("=" * 70)

    # Step 1: Database Initialization
    logger.info("\n--- STEP 1: INITIALIZING SQLITE DATABASE ---")
    init_db()

    # Step 2: Data Preprocessing & Feature Engineering
    logger.info("\n--- STEP 2: PREPROCESSING DATASET & INGESTING SENSOR READINGS ---")
    df = load_and_preprocess_raw_data()
    ingest_dataframe_to_db(df)

    # Step 3: Train Machine Learning Classifier
    logger.info("\n--- STEP 3: TRAINING FAULT CLASSIFIER ---")
    classifier = FaultClassifier()
    metrics = classifier.train_and_evaluate(df)
    logger.info(f"Model Training Complete! Accuracy: {metrics['accuracy']:.4%}, F1 Macro: {metrics['f1_macro']:.4%}")

    # Step 4: Fit Historical Similarity Engine
    logger.info("\n--- STEP 4: TRAINING SIMILARITY SEARCH ENGINE ---")
    similarity_engine = SimilarityEngine(top_k=10)
    similarity_engine.fit(df)

    # Step 5: Process and Index Technical Documentation
    logger.info("\n--- STEP 5: EXTRACTING (OCR) & VECTORIZING TECHNICAL MANUALS ---")
    doc_processor = DocumentProcessor()
    doc_results = doc_processor.process_all_initial_documents()
    logger.info(f"Technical Documents Processed: {json.dumps(doc_results, indent=2)}")

    # Step 6: End-to-End Test with Exam Sample JSON
    logger.info("\n--- STEP 6: VALIDATING END-TO-END INFERENCE ON SAMPLE EVENT ---")
    sample_event = {
        "id": 114387,
        "created_at": "2026-06-01 21:32:53.911176+00:00",
        "z_rms_velocity_in_s": 0.0597,
        "z_rms_velocity_mm_s": 1.517,
        "temperature_f": 76.44,
        "temperature_c": 24.69,
        "x_rms_velocity_in_s": 0.0787,
        "x_rms_velocity_mm_s": 2.0,
        "z_peak_acceleration_g": 0.484,
        "x_peak_acceleration_g": 0.631,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 0.09,
        "x_rms_acceleration_g": 0.114,
        "z_kurtosis": 2.392,
        "x_kurtosis": 2.77,
        "z_crest_factor": 3.747,
        "x_crest_factor": 4.269,
        "z_peak_velocity_in_s": 0.0844,
        "z_peak_velocity_mm_s": 2.146,
        "x_peak_velocity_in_s": 0.1113,
        "x_peak_velocity_mm_s": 2.829,
        "z_high_freq_rms_accel_g": 0.129,
        "x_high_freq_rms_accel_g": 0.147,
        "fault": "cocked_rotor_2",
        "rpm": 1000.0,
    }

    pred_cat, conf, probas = classifier.predict(sample_event)
    sim_res = similarity_engine.find_similar(sample_event)
    rag = get_rag_engine()
    presc = rag.generate_prescription(sample_event, pred_cat, conf, sim_res)

    logger.info(f"Predicted Category: {pred_cat} (Confidence: {conf:.2f}%)")
    logger.info(f"Top Similar Matches Found: {len(sim_res['top_matches'])}")
    logger.info(f"Prescription Generated Successfully: Has Doc = {presc['has_document']}")
    logger.info(f"Prescription Preview:\n{presc['prescription'][:300]}...")

    elapsed = time.time() - start_total
    logger.info("=" * 70)
    logger.info(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {elapsed:.2f}s!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
