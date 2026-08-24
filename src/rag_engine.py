"""
Prescriptive RAG Engine with Google Gemini API, Guardrails, and Context Synthesis.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai

import config
from src.constants import DOCUMENT_MAPPING
from src.document_processor import get_document_processor
from src.database import get_db_connection, log_prescription

logger = logging.getLogger(__name__)


class PrescriptiveRAGEngine:
    """Orchestrates prescriptive maintenance reasoning using Gemini LLM and retrieved documents."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", config.GEMINI_API_KEY)
        self.client_configured = False
        self._configure_client()

    def _configure_client(self) -> None:
        """Initializes Google Generative AI client if key is available."""
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.client_configured = True
                logger.info("Google Gemini API client successfully configured.")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini API client: {e}")
                self.client_configured = False
        else:
            logger.info("No GEMINI_API_KEY found. RAG engine will run in deterministic expert mode.")

    def set_api_key(self, key: str) -> None:
        """Updates the Gemini API key dynamically at runtime."""
        self.api_key = key
        self._configure_client()

    def check_document_availability(self, fault_category: str) -> Dict[str, Any]:
        """Verifies if the fault category has technical documentation registered in SQLite."""
        conn = get_db_connection()
        row = conn.execute(
            "SELECT * FROM fault_documents WHERE fault_category = ?", (fault_category,)
        ).fetchone()
        conn.close()

        if row:
            return {
                "has_document": True,
                "document_name": row["document_name"],
                "title": row["title"],
                "chunk_count": row["chunk_count"],
            }
        
        # Check static mapping
        doc_info = DOCUMENT_MAPPING.get(fault_category, {})
        has_doc = doc_info.get("has_document", False)
        return {
            "has_document": has_doc,
            "document_name": doc_info.get("doc_id"),
            "title": doc_info.get("title", "Sem Documentação"),
            "chunk_count": 0,
        }

    def generate_prescription(
        self,
        event_data: dict,
        predicted_category: str,
        confidence: float,
        similarity_results: dict,
    ) -> Dict[str, Any]:
        """
        Main prescriptive entry point for an incoming sensor event.
        Enforces guardrails: strictly prescribes ONLY for documented faults.
        """
        doc_status = self.check_document_availability(predicted_category)

        # Operational State (Normal)
        if predicted_category == "normal":
            prescription_text = (
                "### ✅ Estado Operacional: Normal\n\n"
                "Os parâmetros estatísticos e de vibração indicam que o equipamento está operando dentro das faixas nominais seguras.\n"
                "- **Recomendação**: Manter o plano de monitoramento preditivo de rotina.\n"
                "- **Ação**: Nenhuma intervenção corretiva necessária no momento."
            )
            log_prescription(
                event_id=event_data.get("id"),
                fault_category=predicted_category,
                confidence=confidence,
                similar_count=similarity_results.get("similar_count", 0),
                prescription_text=prescription_text,
                has_document=False,
                raw_event_json=event_data,
            )
            return {
                "has_document": False,
                "is_operational_normal": True,
                "prescription": prescription_text,
                "fault_category": predicted_category,
                "confidence": confidence,
                "referenced_docs": [],
            }

        # Guardrail: Fault has NO documentation
        if not doc_status["has_document"]:
            missing_text = (
                f"### ⚠️ Defeito Identificado: {predicted_category.replace('_', ' ').title()} (Sem Procedimento Cadastrado)\n\n"
                f"O sistema de Inteligência Artificial diagnosticou o padrão vibracional como **{predicted_category}** "
                f"com **{confidence:.1f}%** de confiança e localizou **{similarity_results.get('total_historical_occurrences', 0):,}** "
                f"ocorrências históricas semelhantes.\n\n"
                f"> **Atenção:** Ainda **não existe documentação técnica ou manual cadastrado** para esta categoria de falha na base corporativa.\n\n"
                f"#### 📋 Ação Recomendada:\n"
                f"1. A equipe técnica/especialista deve registrar e anexar um novo documento orientativo (PDF) na aba de **Documentos**.\n"
                f"2. Realizar inspeção preventiva manual no maquinário para avaliar o componente.\n"
                f"3. Após o upload do procedimento, o modelo passará a prescrever a correção automaticamente."
            )
            log_prescription(
                event_id=event_data.get("id"),
                fault_category=predicted_category,
                confidence=confidence,
                similar_count=similarity_results.get("similar_count", 0),
                prescription_text=missing_text,
                has_document=False,
                raw_event_json=event_data,
            )
            return {
                "has_document": False,
                "is_operational_normal": False,
                "prescription": missing_text,
                "fault_category": predicted_category,
                "confidence": confidence,
                "referenced_docs": [],
            }

        # Fault HAS documentation -> Retrieve chunks and generate prescription
        dp = get_document_processor()
        query = f"Procedimento de correção e diagnóstico para defeito de {predicted_category} vibração temperatura rotação"
        retrieved_chunks = dp.query_chunks(query_text=query, fault_category=predicted_category, n_results=4)

        context_text = "\n\n".join([f"[Trecho {i+1}]: {c['text']}" for i, c in enumerate(retrieved_chunks)])

        # Sensor summary
        sensor_summary = f"""
- ID do Evento: {event_data.get('id', 'N/A')}
- Rotação: {event_data.get('rpm', 0.0)} RPM
- Temperatura: {event_data.get('temperature_c', 0.0)} °C ({event_data.get('temperature_f', 0.0)} °F)
- Vibração RMS Z: {event_data.get('z_rms_velocity_mm_s', 0.0)} mm/s | X: {event_data.get('x_rms_velocity_mm_s', 0.0)} mm/s
- Aceleração Pico Z: {event_data.get('z_peak_acceleration_g', 0.0)} g | X: {event_data.get('x_peak_acceleration_g', 0.0)} g
- Kurtosis Z: {event_data.get('z_kurtosis', 0.0)} | X: {event_data.get('x_kurtosis', 0.0)}
- Crest Factor Z: {event_data.get('z_crest_factor', 0.0)} | X: {event_data.get('x_crest_factor', 0.0)}
        """.strip()

        # Operational historical context
        hist_context = similarity_results.get("operational_context", {})
        hist_summary = f"""
- Eventos similares no histórico: {similarity_results.get('similar_count', 0)}
- Total histórico nesta categoria: {similarity_results.get('total_historical_occurrences', 0):,}
- RPM médio histórico: {hist_context.get('avg_rpm', 0.0)} RPM
- Temperatura média histórica: {hist_context.get('avg_temp_c', 0.0)} °C
        """.strip()

        if self.client_configured:
            try:
                prescription_text = self._call_gemini_prescription(
                    fault_category=predicted_category,
                    confidence=confidence,
                    sensor_summary=sensor_summary,
                    hist_summary=hist_summary,
                    context_text=context_text,
                    doc_title=doc_status.get("title", doc_status.get("document_name", "Manual"))
                )
            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}. Falling back to deterministic RAG synthesis.")
                prescription_text = self._deterministic_rag_synthesis(
                    fault_category=predicted_category,
                    confidence=confidence,
                    sensor_summary=sensor_summary,
                    hist_summary=hist_summary,
                    retrieved_chunks=retrieved_chunks,
                    doc_title=doc_status.get("title", "")
                )
        else:
            prescription_text = self._deterministic_rag_synthesis(
                fault_category=predicted_category,
                confidence=confidence,
                sensor_summary=sensor_summary,
                hist_summary=hist_summary,
                retrieved_chunks=retrieved_chunks,
                doc_title=doc_status.get("title", "")
            )

        log_prescription(
            event_id=event_data.get("id"),
            fault_category=predicted_category,
            confidence=confidence,
            similar_count=similarity_results.get("similar_count", 0),
            prescription_text=prescription_text,
            has_document=True,
            raw_event_json=event_data,
        )

        return {
            "has_document": True,
            "is_operational_normal": False,
            "prescription": prescription_text,
            "fault_category": predicted_category,
            "confidence": confidence,
            "referenced_docs": [doc_status["document_name"]],
            "chunks_used": len(retrieved_chunks),
        }

    def _call_gemini_prescription(
        self,
        fault_category: str,
        confidence: float,
        sensor_summary: str,
        hist_summary: str,
        context_text: str,
        doc_title: str
    ) -> str:
        """Calls Gemini 2.0 Flash with system prompt and retrieved technical chunks."""
        model = genai.GenerativeModel(
            model_name=config.LLM_MODEL,
            system_instruction=(
                "Você é um Especialista Sênior em Manutenção Prescritiva e Confiabilidade Industrial. "
                "Sua função é gerar recomendações prescritivas precisas, práticas e estruturadas com base "
                "EXCLUSIVAMENTE nos manuais e procedimentos técnicos fornecidos. "
                "Nunca invente procedimentos fora da documentação fornecida. Estruture a resposta em tópicos claros "
                "incluindo: 1. Diagnóstico e Severidade; 2. Procedimento Passo a Passo de Correção; "
                "3. Ferramentas e Instrumentos Necessários; 4. Critérios de Aceitação e Segurança; 5. Recomendações Preventivas."
            )
        )

        prompt = f"""
## Evento Monitorado
- Diagnóstico IA: **{fault_category.upper().replace('_', ' ')}** (Confiança: {confidence:.1f}%)
- Documento Referência: {doc_title}

### Dados do Sensor em Tempo Real:
{sensor_summary}

### Padrão Histórico e Contexto Operacional:
{hist_summary}

### Procedimento Técnico Extraído da Documentação Oficial:
{context_text}

Gere a prescrição técnica detalhada e objetiva para a equipe de manutenção de chão de fábrica.
"""
        response = model.generate_content(prompt)
        return response.text

    def _deterministic_rag_synthesis(
        self,
        fault_category: str,
        confidence: float,
        sensor_summary: str,
        hist_summary: str,
        retrieved_chunks: List[Dict[str, Any]],
        doc_title: str
    ) -> str:
        """Deterministic offline fallback synthesis that extracts instructions directly from document chunks."""
        sections = []
        sections.append(f"### 🛠️ Prescrição Técnica: {fault_category.replace('_', ' ').title()}")
        sections.append(f"**Diagnóstico de IA:** {fault_category} (Confiança: **{confidence:.1f}%**)\n")
        sections.append(f"**Manual de Referência:** *{doc_title}*\n")
        
        sections.append("#### 📊 Contexto do Evento e Histórico:")
        sections.append(f"```\n{sensor_summary}\n\n{hist_summary}\n```\n")

        sections.append("#### 📖 Procedimentos e Ações Extraídas dos Manuais Técnicos:")
        for idx, chunk in enumerate(retrieved_chunks):
            clean_chunk = chunk["text"].replace("--- Página", "\n*Página").strip()
            sections.append(f"> **Trecho Técnico {idx+1}:**\n{clean_chunk}\n")

        sections.append("#### 🛡️ Medidas de Segurança Obrigatórias:")
        sections.append("1. Desligar e desenergizar o motor/equipamento antes de qualquer intervenção.")
        sections.append("2. Aplicar procedimento padrão de Bloqueio e Etiquetagem (LOTO).")
        sections.append("3. Utilizar EPIs adequados (óculos, luvas, calçado de segurança, protetor auricular).")
        sections.append("4. Confirmar parada total das partes girantes antes da inspeção mecânica.")

        return "\n\n".join(sections)

    def chat_with_technician(self, user_message: str, chat_history: list, current_event_context: Optional[dict] = None) -> str:
        """Conversational chat interface with strict guardrails to discuss documented maintenance procedures."""
        if not self.client_configured:
            # Fallback response if no Gemini API Key is provided
            return (
                "Para conversar interativamente com o assistente IA em linguagem natural, "
                "por favor configure a sua **GEMINI_API_KEY** na barra lateral ou no arquivo `.env`. "
                "Enquanto isso, as prescrições completas e extração dos manuais estão disponíveis na aba **Novo Evento**."
            )

        model = genai.GenerativeModel(
            model_name=config.LLM_MODEL,
            system_instruction=(
                "Você é o Assistente Virtual Especialista em Manutenção Prescritiva Industrial. "
                "Seu escopo é ESTRITAMENTE responder dúvidas sobre a operação, diagnóstico e procedimentos técnicos "
                "de manutenção dos equipamentos monitorados (cobertos pelos manuais técnicos de rolamentos, desalinhamento, "
                "desbalanceamento, correias, polias e rotor inclinado). "
                "Se o usuário perguntar sobre assuntos fora desse domínio industrial ou sobre falhas que não possuem documento, "
                "informe gentilmente que suas orientações são restritas aos manuais técnicos cadastrados na planta industrial."
            )
        )

        history_prompts = []
        for msg in chat_history[-6:]:
            role = "user" if msg["role"] == "user" else "model"
            history_prompts.append({"role": role, "parts": [msg["content"]]})

        context_prefix = ""
        if current_event_context:
            context_prefix = f"[Contexto do Evento Atual: Defeito={current_event_context.get('fault_category')}, Confiança={current_event_context.get('confidence')}%]\n"

        chat = model.start_chat(history=history_prompts[:-1] if len(history_prompts) > 1 else [])
        response = chat.send_message(context_prefix + user_message)
        return response.text


# Global RAG engine singleton
rag_engine_instance = PrescriptiveRAGEngine()


def get_rag_engine() -> PrescriptiveRAGEngine:
    """Returns the singleton RAG engine instance."""
    return rag_engine_instance
