"""
Unit tests for Historical Similarity Engine.
"""

import pytest
from src.similarity import get_similarity_engine


def test_similarity_search():
    similarity_engine = get_similarity_engine()
    assert similarity_engine.nn_model is not None, "Similarity engine should be fitted"

    sample_event = {
        "id": 114387,
        "rpm": 1000.0,
        "temperature_c": 24.69,
        "z_rms_velocity_mm_s": 1.517,
        "x_rms_velocity_mm_s": 2.0,
        "z_peak_acceleration_g": 0.484,
        "x_peak_acceleration_g": 0.631,
        "z_rms_acceleration_g": 0.09,
        "x_rms_acceleration_g": 0.114,
        "z_kurtosis": 2.392,
        "x_kurtosis": 2.77,
        "z_crest_factor": 3.747,
        "x_crest_factor": 4.269,
        "z_high_freq_rms_accel_g": 0.129,
        "x_high_freq_rms_accel_g": 0.147,
    }

    res = similarity_engine.find_similar(sample_event, top_k=5)
    assert res["similar_count"] == 5
    assert len(res["top_matches"]) == 5
    assert "operational_context" in res
    assert "time_distribution" in res
    assert res["top_matches"][0]["similarity_score"] > 50.0
