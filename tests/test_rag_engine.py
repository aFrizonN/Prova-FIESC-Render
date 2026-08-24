"""
Unit tests for Prescriptive RAG Engine and Guardrails.
"""

import pytest
from src.rag_engine import get_rag_engine


def test_rag_guardrail_documented_fault():
    rag = get_rag_engine()
    event_data = {"id": 999, "rpm": 1000.0, "temperature_c": 25.0}
    sim_res = {"similar_count": 5, "total_historical_occurrences": 1000, "operational_context": {"avg_rpm": 1000.0, "avg_temp_c": 25.0}}

    # Test with documented fault (desalinhamento -> Doc2)
    res = rag.generate_prescription(
        event_data=event_data,
        predicted_category="desalinhamento",
        confidence=95.0,
        similarity_results=sim_res
    )
    assert res["has_document"] is True
    assert "desalinhamento" in res["prescription"].lower() or "desalinhado" in res["prescription"].lower()
    assert len(res["referenced_docs"]) > 0


def test_rag_guardrail_undocumented_fault():
    rag = get_rag_engine()
    event_data = {"id": 998, "rpm": 1000.0, "temperature_c": 25.0}
    sim_res = {"similar_count": 5, "total_historical_occurrences": 500, "operational_context": {"avg_rpm": 1000.0, "avg_temp_c": 25.0}}

    # Test with undocumented fault (eccentric_rotor)
    res = rag.generate_prescription(
        event_data=event_data,
        predicted_category="eccentric_rotor",
        confidence=92.0,
        similarity_results=sim_res
    )
    assert res["has_document"] is False
    assert "Ainda **não existe documentação técnica ou manual cadastrado**" in res["prescription"]
    assert "registrar e anexar um novo documento" in res["prescription"]


def test_rag_operational_normal():
    rag = get_rag_engine()
    event_data = {"id": 997, "rpm": 1000.0, "temperature_c": 23.0}
    sim_res = {"similar_count": 5, "total_historical_occurrences": 5000, "operational_context": {}}

    res = rag.generate_prescription(
        event_data=event_data,
        predicted_category="normal",
        confidence=99.0,
        similarity_results=sim_res
    )
    assert res["is_operational_normal"] is True
    assert "Estado Operacional: Normal" in res["prescription"]
