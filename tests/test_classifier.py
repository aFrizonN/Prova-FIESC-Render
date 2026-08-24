"""
Unit tests for Machine Learning Classifier.
"""

import pytest
from src.classifier import get_classifier


def test_classifier_predict_sample():
    classifier = get_classifier()
    assert classifier.model is not None, "Model should be trained and loaded"
    
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
        "rpm": 1000.0,
    }
    
    pred_cat, conf, probas = classifier.predict(sample_event)
    assert isinstance(pred_cat, str)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 100.0
    assert len(probas) > 0
    assert pred_cat == "cocked_rotor"
