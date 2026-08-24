"""
Unit tests for data preprocessing, category mapping, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np

from src.constants import map_fault_to_category, is_operational_state, SENSOR_FEATURES
from src.data_processing import extract_features, prepare_single_event_features, ALL_MODEL_FEATURES


def test_map_fault_to_category():
    # Normal / states
    assert map_fault_to_category("normal") == "normal"
    assert map_fault_to_category("motor_desligado") == "normal"
    assert map_fault_to_category("new_teste") == "normal"
    assert is_operational_state("normal_2") is True
    
    # Faults
    assert map_fault_to_category("desalinhado_2") == "desalinhamento"
    assert map_fault_to_category("desbalanceado_1parafuso") == "desbalanceamento"
    assert map_fault_to_category("cocked_rotor_2") == "cocked_rotor"
    assert map_fault_to_category("rolamento_outer_2") == "rolamento_pista_externa"
    assert map_fault_to_category("rolamento_inner_carga") == "rolamento_pista_interna"
    assert map_fault_to_category("rolamento_ball_4") == "rolamento_elementos_rolantes"
    assert map_fault_to_category("correia_2") == "correia"
    assert map_fault_to_category("polia") == "polia"
    assert map_fault_to_category("eccentric_rotor_3") == "eccentric_rotor"
    assert is_operational_state("cocked_rotor") is False


def test_feature_engineering():
    sample_df = pd.DataFrame([{
        "z_rms_velocity_mm_s": 2.0,
        "x_rms_velocity_mm_s": 1.0,
        "z_peak_acceleration_g": 0.5,
        "x_peak_acceleration_g": 0.25,
        "z_rms_acceleration_g": 0.1,
        "x_rms_acceleration_g": 0.1,
        "z_kurtosis": 3.0,
        "x_kurtosis": 3.0,
        "z_crest_factor": 4.0,
        "x_crest_factor": 4.0,
        "z_high_freq_rms_accel_g": 0.2,
        "x_high_freq_rms_accel_g": 0.1,
    }])
    
    feat_df = extract_features(sample_df)
    assert "vel_ratio_zx" in feat_df.columns
    assert "total_vibration_energy" in feat_df.columns
    assert np.isclose(feat_df["vel_ratio_zx"].iloc[0], 2.0, atol=1e-3)
    assert np.isclose(feat_df["acc_ratio_zx"].iloc[0], 2.0, atol=1e-3)


def test_prepare_single_event_features():
    event = {
        "rpm": 1000.0,
        "temperature_c": 25.0,
        "z_rms_velocity_mm_s": 1.5,
        "x_rms_velocity_mm_s": 2.0,
        "z_peak_acceleration_g": 0.5,
        "x_peak_acceleration_g": 0.6,
        "z_rms_acceleration_g": 0.1,
        "x_rms_acceleration_g": 0.1,
        "z_kurtosis": 2.5,
        "x_kurtosis": 2.8,
        "z_crest_factor": 3.5,
        "x_crest_factor": 4.0,
        "z_high_freq_rms_accel_g": 0.1,
        "x_high_freq_rms_accel_g": 0.15,
    }
    
    prepared = prepare_single_event_features(event)
    assert list(prepared.columns) == ALL_MODEL_FEATURES
    assert len(prepared) == 1
