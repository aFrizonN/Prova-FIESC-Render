"""
Streamlit Prescriptive Maintenance Web Application.
Interactive Industrial Dashboard, Telemetry Analysis, Prescriptive RAG, and Technical Chat.
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import config
from src.constants import DOCUMENT_MAPPING, SENSOR_FEATURES
from src.classifier import get_classifier
from src.similarity import get_similarity_engine
from src.rag_engine import get_rag_engine
from src.document_processor import get_document_processor
from src.database import get_kpis, get_fault_distribution, get_time_series_faults, get_db_connection, init_db

# Page configuration
st.set_page_config(
    page_title="Prescriptive Maintenance AI | SENAI SC",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-initialize database tables if not yet present
init_db()

# Custom CSS for Industrial UI with crisp contrast
st.markdown("""
<style>
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        color: #f8fafc !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8 !important;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #cbd5e1 !important;
    }
    
    /* Result Box */
    .diagnosis-card {
        background: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        color: #f8fafc;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Sample Presets for Testing
SAMPLE_PRESETS = {
    "Exame (Cocked Rotor 2)": {
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
        "rpm": 1000.0
    },
    "Desalinhamento": {
        "id": 54201,
        "created_at": "2026-05-12 14:10:00.000000+00:00",
        "z_rms_velocity_in_s": 0.092,
        "z_rms_velocity_mm_s": 2.337,
        "temperature_f": 78.5,
        "temperature_c": 25.83,
        "x_rms_velocity_in_s": 0.125,
        "x_rms_velocity_mm_s": 3.175,
        "z_peak_acceleration_g": 0.85,
        "x_peak_acceleration_g": 1.12,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 0.18,
        "x_rms_acceleration_g": 0.22,
        "z_kurtosis": 3.1,
        "x_kurtosis": 3.45,
        "z_crest_factor": 4.2,
        "x_crest_factor": 4.8,
        "z_peak_velocity_in_s": 0.13,
        "z_peak_velocity_mm_s": 3.3,
        "x_peak_velocity_in_s": 0.18,
        "x_peak_velocity_mm_s": 4.57,
        "z_high_freq_rms_accel_g": 0.19,
        "x_high_freq_rms_accel_g": 0.24,
        "fault": "desalinhado",
        "rpm": 1500.0
    },
    "Desbalanceamento": {
        "id": 31204,
        "created_at": "2026-05-08 09:30:00.000000+00:00",
        "z_rms_velocity_in_s": 0.115,
        "z_rms_velocity_mm_s": 2.921,
        "temperature_f": 79.1,
        "temperature_c": 26.17,
        "x_rms_velocity_in_s": 0.145,
        "x_rms_velocity_mm_s": 3.683,
        "z_peak_acceleration_g": 0.95,
        "x_peak_acceleration_g": 1.35,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 0.21,
        "x_rms_acceleration_g": 0.28,
        "z_kurtosis": 2.65,
        "x_kurtosis": 2.89,
        "z_crest_factor": 3.8,
        "x_crest_factor": 4.1,
        "z_peak_velocity_in_s": 0.16,
        "z_peak_velocity_mm_s": 4.06,
        "x_peak_velocity_in_s": 0.21,
        "x_peak_velocity_mm_s": 5.33,
        "z_high_freq_rms_accel_g": 0.22,
        "x_high_freq_rms_accel_g": 0.31,
        "fault": "desbalanceado_1parafuso",
        "rpm": 2000.0
    },
    "Falha Sem Documento (Rotor Excêntrico)": {
        "id": 89012,
        "created_at": "2026-05-20 16:45:00.000000+00:00",
        "z_rms_velocity_in_s": 0.088,
        "z_rms_velocity_mm_s": 2.235,
        "temperature_f": 82.0,
        "temperature_c": 27.78,
        "x_rms_velocity_in_s": 0.118,
        "x_rms_velocity_mm_s": 2.997,
        "z_peak_acceleration_g": 0.72,
        "x_peak_acceleration_g": 0.98,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 0.16,
        "x_rms_acceleration_g": 0.19,
        "z_kurtosis": 2.8,
        "x_kurtosis": 3.05,
        "z_crest_factor": 3.9,
        "x_crest_factor": 4.3,
        "z_peak_velocity_in_s": 0.12,
        "z_peak_velocity_mm_s": 3.05,
        "x_peak_velocity_in_s": 0.17,
        "x_peak_velocity_mm_s": 4.32,
        "z_high_freq_rms_accel_g": 0.18,
        "x_high_freq_rms_accel_g": 0.21,
        "fault": "eccentric_rotor",
        "rpm": 1200.0
    },
    "Operação Normal (Baseline)": {
        "id": 1426,
        "created_at": "2026-04-30 17:17:41.549800+00:00",
        "z_rms_velocity_in_s": 0.0427,
        "z_rms_velocity_mm_s": 1.086,
        "temperature_f": 74.0,
        "temperature_c": 23.33,
        "x_rms_velocity_in_s": 0.0619,
        "x_rms_velocity_mm_s": 1.573,
        "z_peak_acceleration_g": 0.031,
        "x_peak_acceleration_g": 0.033,
        "z_peak_vel_comp_freq_hz": 61.0,
        "x_peak_vel_comp_freq_hz": 61.0,
        "z_rms_acceleration_g": 0.046,
        "x_rms_acceleration_g": 0.066,
        "z_kurtosis": 3.276,
        "x_kurtosis": 3.25,
        "z_crest_factor": 4.435,
        "x_crest_factor": 3.918,
        "z_peak_velocity_in_s": 0.0605,
        "z_peak_velocity_mm_s": 1.536,
        "x_peak_velocity_in_s": 0.0875,
        "x_peak_velocity_mm_s": 2.224,
        "z_high_freq_rms_accel_g": 0.007,
        "x_high_freq_rms_accel_g": 0.008,
        "fault": "normal",
        "rpm": 1000.0
    }
}


# Sidebar Navigation & Configuration
with st.sidebar:
    st.title("🏭 Prescritivo IA")
    st.caption("Plataforma Industrial de Confiabilidade")
    st.markdown("---")

    # API Configuration
    st.subheader("🔑 Configuração Gemini")
    api_key_input = st.text_input(
        "Google Gemini API Key:",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Insira sua chave do Google AI Studio para ativar o RAG conversacional com Gemini 2.0 Flash."
    )
    if api_key_input:
        get_rag_engine().set_api_key(api_key_input)
        st.success("Gemini API Ativa! 🟢")
    else:
        st.info("Modo Offline / RAG Determinístico Ativo 🟡")

    st.markdown("---")
    st.subheader("📡 Status do Sistema")
    classifier = get_classifier()
    model_loaded = classifier.model is not None
    st.write(f"• **Classificador ML:** {'🟢 Carregado (RF 84.2%)' if model_loaded else '🟡 Inicializando...'}")
    st.write("• **Busca Similaridade:** 🟢 Ativo")
    st.write("• **Vector Store (ChromaDB):** 🟢 Ativo")
    st.write("• **Banco SQLite:** 🟢 Conectado")

    st.markdown("---")
    st.caption("SENAI SC | Manutenção Prescritiva")


# Header
st.title("🏭 Sistema Integrado de Manutenção Prescritiva com IA")
st.markdown("Monitoramento de telemetria vibracional, diagnóstico preditivo de anomalias e prescrição automatizada com base em procedimentos técnicos industriais.")

# Main Navigation Tabs
tab_dash, tab_event, tab_chat, tab_docs = st.tabs([
    "📊 Dashboard Operacional",
    "🔍 Análise & Prescrição (Novo Evento)",
    "💬 Assistente Técnico (Chat)",
    "📄 Gestão de Documentos (RAG)"
])


# ==============================================================================
# TAB 1: DASHBOARD OPERACIONAL
# ==============================================================================
with tab_dash:
    st.subheader("Painel de Controle em Tempo Real")
    
    # KPIs
    kpis = get_kpis()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total de Eventos Monitorados</div>
            <div class="metric-value">{kpis['total_events']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Anomalias / Falhas Registradas</div>
            <div class="metric-value" style="color: #f87171 !important;">{kpis['total_problems']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Manuais Técnicos Cadastrados</div>
            <div class="metric-value" style="color: #34d399 !important;">{kpis['total_docs']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Prescrições Geradas</div>
            <div class="metric-value" style="color: #a78bfa !important;">{kpis['total_prescriptions']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("##### 📌 Distribuição de Ocorrências por Categoria")
        df_dist = get_fault_distribution()
        if not df_dist.empty:
            fig_dist = px.pie(
                df_dist,
                names="fault_category",
                values="count",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Dark24,
            )
            fig_dist.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Nenhum dado registrado para exibir no gráfico.")

    with col_chart2:
        st.markdown("##### 📈 Volume de Ocorrências por Tipo de Defeito")
        if not df_dist.empty:
            df_prob = df_dist[df_dist["is_problem"] == 1].sort_values("count", ascending=True)
            if not df_prob.empty:
                fig_bar = px.bar(
                    df_prob,
                    x="count",
                    y="fault_category",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Viridis",
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Número de Leituras",
                    yaxis_title="Defeito",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Nenhuma falha registrada.")
        else:
            st.info("Nenhum dado registrado.")

    # Timeline Row
    st.markdown("##### ⏱️ Linha Temporal de Detecções de Anomalias")
    df_time = get_time_series_faults()
    if not df_time.empty:
        fig_time = px.area(
            df_time,
            x="date",
            y="occurrences",
            color="fault_category",
            template="plotly_dark",
        )
        fig_time.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Data de Aquisição",
            yaxis_title="Contagem de Eventos",
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Aguardando novas leituras para gerar a série temporal.")


# ==============================================================================
# TAB 2: ANÁLISE & PRESCRIÇÃO (NOVO EVENTO)
# ==============================================================================
with tab_event:
    st.subheader("Entrada de Telemetria e Diagnóstico Prescritivo")
    st.markdown("Envie uma leitura de vibração em tempo real (JSON) para execução automática do pipeline de IA.")

    # Preset selection buttons
    st.markdown("**Carregar Exemplo de Teste:**")
    cols_btn = st.columns(len(SAMPLE_PRESETS))
    for idx, (p_name, p_data) in enumerate(SAMPLE_PRESETS.items()):
        if cols_btn[idx].button(p_name, key=f"btn_preset_{idx}"):
            st.session_state["input_json_text"] = json.dumps(p_data, indent=2)

    default_json = st.session_state.get(
        "input_json_text",
        json.dumps(SAMPLE_PRESETS["Exame (Cocked Rotor 2)"], indent=2)
    )

    json_input = st.text_area(
        "JSON de Entrada do Sensor:",
        value=default_json,
        height=180,
        help="Cole aqui o payload JSON transmitido pelos sensores de vibração e temperatura."
    )

    if st.button("🚀 Processar Evento & Prescrever Ação", type="primary", use_container_width=True):
        try:
            event_data = json.loads(json_input)
            st.session_state["current_analyzed_event"] = event_data

            with st.spinner("Executando classificação ML, busca por similaridade e recuperação de conhecimento..."):
                classifier = get_classifier()
                similarity_engine = get_similarity_engine()
                rag_engine = get_rag_engine()

                # Step 1: Predict
                predicted_category, confidence, probas = classifier.predict(event_data)
                
                # Step 2: Similarity
                sim_res = similarity_engine.find_similar(event_data, top_k=10)
                
                # Step 3: Prescribe
                presc_res = rag_engine.generate_prescription(
                    event_data=event_data,
                    predicted_category=predicted_category,
                    confidence=confidence,
                    similarity_results=sim_res
                )

                st.session_state["last_prescription_result"] = presc_res
                st.session_state["last_similarity_result"] = sim_res

            st.success("Diagnóstico e Prescrição processados com sucesso!")

            # Display Results
            res_c1, res_c2 = st.columns([1.2, 1])

            with res_c1:
                st.markdown("### 🎯 Diagnóstico da Inteligência Artificial")
                
                badge_color = "#10b981" if predicted_category == "normal" else ("#38bdf8" if presc_res["has_document"] else "#f97316")
                st.markdown(f"""
                <div class="diagnosis-card">
                    <div style="font-size: 0.9rem; color: #94a3b8;">DEFEITO IDENTIFICADO</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {badge_color};">
                        {predicted_category.upper().replace('_', ' ')}
                    </div>
                    <div style="margin-top: 8px; color: #cbd5e1;">
                        Confiança do Modelo: <strong>{confidence:.1f}%</strong> | 
                        Documentação: <strong>{'✅ Cadastrada' if presc_res['has_document'] else ('ℹ️ Operação Normal' if predicted_category == 'normal' else '⚠️ Não Encontrada')}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Prescrição Técnica & Ações de Manutenção")
                st.markdown(presc_res["prescription"])

            with res_c2:
                st.markdown("### 📊 Distribuição de Probabilidade ML")
                top_5_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)[:6]
                df_prob_plot = pd.DataFrame(top_5_probas, columns=["Categoria", "Probabilidade (%)"])
                fig_prob = px.bar(
                    df_prob_plot,
                    x="Probabilidade (%)",
                    y="Categoria",
                    orientation="h",
                    color="Probabilidade (%)",
                    color_continuous_scale="Teal",
                )
                fig_prob.update_layout(
                    template="plotly_dark",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=240,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                st.markdown("### 🔄 Ocorrências Similares no Histórico")
                op_ctx = sim_res["operational_context"]
                st.markdown(f"""
                - **Eventos Similares Identificados:** {sim_res['similar_count']} registros
                - **Total Histórico da Categoria:** {sim_res['total_historical_occurrences']:,}
                - **Faixa de RPM Observada:** {op_ctx['rpm_range'][0]} - {op_ctx['rpm_range'][1]} RPM (Média: {op_ctx['avg_rpm']} RPM)
                - **Faixa de Temperatura:** {op_ctx['temp_c_range'][0]} - {op_ctx['temp_c_range'][1]} °C
                - **Vibração RMS Média (Z / X):** {op_ctx['avg_z_rms_velocity']} mm/s / {op_ctx['avg_x_rms_velocity']} mm/s
                """)

                # Similar Events Table
                st.markdown("##### 🔍 Top 5 Registros Mais Próximos:")
                top_matches = sim_res["top_matches"][:5]
                if top_matches:
                    df_matches = pd.DataFrame(top_matches)[["id", "similarity_score", "fault", "rpm", "temperature_c"]]
                    df_matches.columns = ["ID", "Similaridade (%)", "Rótulo Original", "RPM", "Temp (°C)"]
                    st.dataframe(df_matches, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Erro ao processar JSON: {e}")


# ==============================================================================
# TAB 3: ASSISTENTE TÉCNICO (CHAT)
# ==============================================================================
with tab_chat:
    st.subheader("Assistente IA Especialista em Manutenção")
    st.caption("Tire dúvidas sobre os procedimentos técnicos de manutenção, ferramentas, medições e normas de segurança.")

    # Guardrail badge
    st.info("🛡️ **Guardrail Ativo:** O assistente responde estritamente sobre as falhas cobertas pelos manuais técnicos cadastrados.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Olá! Sou seu assistente de Manutenção Prescritiva. Como posso ajudar na intervenção ou diagnóstico do maquinário?"}
        ]

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Digite sua dúvida técnica... (ex: Qual o procedimento para corrigir pé manco no alinhamento?)"):
        st.session_state["chat_messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando manuais técnicos..."):
                rag_engine = get_rag_engine()
                current_ctx = st.session_state.get("last_prescription_result")
                assistant_reply = rag_engine.chat_with_technician(
                    user_message=user_prompt,
                    chat_history=st.session_state["chat_messages"],
                    current_event_context=current_ctx
                )
                st.markdown(assistant_reply)

        st.session_state["chat_messages"].append({"role": "assistant", "content": assistant_reply})


# ==============================================================================
# TAB 4: GESTÃO DE DOCUMENTOS (RAG)
# ==============================================================================
with tab_docs:
    st.subheader("Base Conhecimento & Procedimentos Técnicos")
    st.markdown("Gerencie a base documental utilizada pelo sistema RAG para gerar prescrições.")

    col_doc_list, col_doc_upload = st.columns([1.2, 1])

    with col_doc_list:
        st.markdown("#### 📚 Manuais Cadastrados no Sistema")
        conn = get_db_connection()
        docs_df = pd.read_sql_query("SELECT fault_category, document_name, title, chunk_count, uploaded_at FROM fault_documents", conn)
        conn.close()

        if not docs_df.empty:
            docs_df.columns = ["Categoria de Falha", "Arquivo PDF", "Título do Procedimento", "Chunks", "Última Atualização"]
            st.dataframe(docs_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Nenhum documento cadastrado no banco de dados.")

        st.markdown("#### 🔍 Testar Busca Semântica no Vector Store (ChromaDB)")
        search_query = st.text_input("Buscar termo ou procedimento técnico:", "como identificar folga no rolamento")
        if search_query:
            dp = get_document_processor()
            matched = dp.query_chunks(search_query, n_results=3)
            for i, m in enumerate(matched):
                with st.expander(f"Resultado {i+1}: {m['metadata'].get('title', 'Manual')} (Similaridade: {m['score']*100:.1f}%)"):
                    st.markdown(m["text"])

    with col_doc_upload:
        st.markdown("#### 📤 Cadastrar Novo Procedimento Técnico")
        st.markdown("Anexe um novo manual em PDF para associar a um tipo de defeito sem documentação.")

        with st.form("form_upload_doc"):
            new_cat = st.selectbox(
                "Categoria de Falha Associada:",
                ["eccentric_rotor", "ventoinha", "falta_de_fase", "outro_defeito"],
                help="Selecione a falha que este manual orienta a corrigir."
            )
            new_title = st.text_input("Título do Documento:", "Procedimento de Manutenção de Rotor Excêntrico")
            uploaded_pdf = st.file_uploader("Arquivo PDF:", type=["pdf"])
            submit_doc = st.form_submit_button("Indexar no RAG & SQLite")

            if submit_doc and uploaded_pdf:
                with st.spinner("Processando PDF e gerando embeddings..."):
                    dp = get_document_processor()
                    res = dp.add_new_document(
                        uploaded_file_bytes=uploaded_pdf.getvalue(),
                        filename=uploaded_pdf.name,
                        fault_category=new_cat,
                        custom_title=new_title
                    )
                    st.success(f"Documento '{uploaded_pdf.name}' cadastrado e indexado com sucesso ({res['chunks_indexed']} chunks)!")
                    st.rerun()
