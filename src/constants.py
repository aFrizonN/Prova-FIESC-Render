"""
Constants and domain definitions for the Prescriptive Maintenance System.
"""

from typing import Dict, List, Set

# Non-problem operating states
OPERATIONAL_STATES: Set[str] = {
    "normal",
    "normal_2",
    "normal_2_pos_2",
    "normal_3",
    "normal_3_pos_2",
    "normal_6",
    "normal_adxl_0",
    "normal_adxl_1",
    "normal_carga",
    "normal_carga_3",
    "normal_carga_3_2",
    "normal_carga_3_3",
    "normal_novo",
    "normal_novo_teste",
    "normal_pos_2",
    "normla_carga_3_3",
    "baseline",
    "new_baseline",
    "teste",
    "new_tes",
    "new_teste",
    "acelerando",
    "motor_desligado",
    "motor_desligado_novo",
    "mortor_desligado_novo",
    "new_normal_0",
    "new_normal_1",
    "new_normal_2",
    "new_normal_3",
    "new_normal_4",
    "new_normal_5",
    "new_normal_6",
}

# Macro categories mapping
CATEGORY_MAPPING: Dict[str, str] = {
    # Normal / States
    "normal": "normal",
    "baseline": "normal",
    "teste": "normal",
    "acelerando": "normal",
    "motor_desligado": "normal",
    
    # Bearing faults
    "rolamento_outer": "rolamento_pista_externa",
    "new_rolamento_outer": "rolamento_pista_externa",
    "rolamento_inner": "rolamento_pista_interna",
    "new_rolamento_inner": "rolamento_pista_interna",
    "rolamento_ball": "rolamento_elementos_rolantes",
    "new_rolamento_ball": "rolamento_elementos_rolantes",
    "rolamento_combination": "rolamento_combinado",
    "rolamento_comb": "rolamento_combinado",
    "new_rolamento_comb": "rolamento_combinado",
    
    # Misalignment
    "desalinhado": "desalinhamento",
    "new_desalinhado": "desalinhamento",
    
    # Imbalance
    "desbalanceado": "desbalanceamento",
    "desabalanceado": "desbalanceamento",
    "desbanlanceado": "desbalanceamento",
    "ddesbalanceado": "desbalanceamento",
    "dedesbalanceado": "desbalanceamento",
    "new_desbalanceado": "desbalanceamento",
    "new_desabanceado": "desbalanceamento",
    
    # Belts
    "correia": "correia",
    
    # Pulleys
    "polia": "polia",
    
    # Cocked Rotor
    "cocked_rotor": "cocked_rotor",
    "cocked_adxl": "cocked_rotor",
    "cockecocked_adxl": "cocked_rotor",
    "new_cocked": "cocked_rotor",
    
    # Eccentric Rotor (No doc initially)
    "eccentric_rotor": "eccentric_rotor",
    "eccentric_adxl": "eccentric_rotor",
    "eccentric": "eccentric_rotor",
    "new_eccentric": "eccentric_rotor",
    
    # Fan / Ventoinha (No doc initially)
    "ventoinha": "ventoinha",
    
    # Phase fault (No doc initially)
    "new_falta_fase": "falta_de_fase",
}

# Technical documents mapping to macro categories
DOCUMENT_MAPPING: Dict[str, Dict[str, str]] = {
    "rolamento_pista_externa": {
        "doc_id": "Doc1.pdf",
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Rolamentos (Pista Externa)",
        "has_document": True,
    },
    "rolamento_pista_interna": {
        "doc_id": "Doc1.pdf",
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Rolamentos (Pista Interna)",
        "has_document": True,
    },
    "rolamento_elementos_rolantes": {
        "doc_id": "Doc1.pdf",
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Rolamentos (Elementos Rolantes)",
        "has_document": True,
    },
    "rolamento_combinado": {
        "doc_id": "Doc1.pdf",
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Rolamentos (Gaiola / Combinação)",
        "has_document": True,
    },
    "desalinhamento": {
        "doc_id": "Doc2.pdf",
        "title": "Procedimento para Correção de Desalinhamento em Motor Elétrico",
        "has_document": True,
    },
    "desbalanceamento": {
        "doc_id": "Doc3.pdf",
        "title": "Procedimento para Correção de Desbalanceamento em Máquinas Rotativas",
        "has_document": True,
    },
    "correia": {
        "doc_id": "Doc4.pdf",
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Sistemas de Transmissão por Correias",
        "has_document": True,
    },
    "polia": {
        "doc_id": "Doc5.pdf",
        "title": "Procedimento para Diagnóstico e Correção de Problemas em Polias de Sistemas Rotativos",
        "has_document": True,
    },
    "cocked_rotor": {
        "doc_id": "Doc6.pdf",
        "title": "Procedimento para Diagnóstico e Correção de Problemas de Rotor Inclinado (Cocked Rotor)",
        "has_document": True,
    },
    # Fault categories with NO documentation initially:
    "eccentric_rotor": {
        "doc_id": None,
        "title": "Rotor Excêntrico (Sem Documentação Cadastrada)",
        "has_document": False,
    },
    "ventoinha": {
        "doc_id": None,
        "title": "Problemas em Ventoinha / Fluxo de Ar (Sem Documentação Cadastrada)",
        "has_document": False,
    },
    "falta_de_fase": {
        "doc_id": None,
        "title": "Falta de Fase Elétrica (Sem Documentação Cadastrada)",
        "has_document": False,
    },
    "normal": {
        "doc_id": None,
        "title": "Operação Normal do Equipamento",
        "has_document": False,
    },
}

# Sensor feature columns in banner.csv
SENSOR_FEATURES: List[str] = [
    "z_rms_velocity_in_s",
    "z_rms_velocity_mm_s",
    "temperature_f",
    "temperature_c",
    "x_rms_velocity_in_s",
    "x_rms_velocity_mm_s",
    "z_peak_acceleration_g",
    "x_peak_acceleration_g",
    "z_peak_vel_comp_freq_hz",
    "x_peak_vel_comp_freq_hz",
    "z_rms_acceleration_g",
    "x_rms_acceleration_g",
    "z_kurtosis",
    "x_kurtosis",
    "z_crest_factor",
    "x_crest_factor",
    "z_peak_velocity_in_s",
    "z_peak_velocity_mm_s",
    "x_peak_velocity_in_s",
    "x_peak_velocity_mm_s",
    "z_high_freq_rms_accel_g",
    "x_high_freq_rms_accel_g",
    "rpm",
]

def map_fault_to_category(raw_fault: str) -> str:
    """Maps a raw fault string to its macro category."""
    if not isinstance(raw_fault, str):
        return "desconhecido"
    
    clean = raw_fault.strip().lower()
    
    # Direct match in OPERATIONAL_STATES
    if clean in OPERATIONAL_STATES:
        return "normal"
    
    # Prefix-based match
    for prefix, cat in CATEGORY_MAPPING.items():
        if clean.startswith(prefix) or prefix in clean:
            return cat
            
    return "outro_defeito"

def is_operational_state(raw_fault: str) -> bool:
    """Checks if a fault label represents an operational state (not a problem)."""
    return map_fault_to_category(raw_fault) == "normal"
