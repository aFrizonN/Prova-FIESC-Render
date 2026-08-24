# 🏭 Sistema Integrado de Manutenção Prescritiva com IA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-AI_Studio-blueviolet.svg)](https://aistudio.google.com/)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](tests/)

Solução completa de **Inteligência Artificial para Manutenção Prescritiva Industrial**, desenvolvida para o desafio de inovação do **SENAI SC** para aplicação em chão de fábrica de máquinas rotativas (motores elétricos, mancais, transmissões por correias e polias).

---

## 📌 1. Visão Geral e Objetivos do Projeto

A solução vai além da manutenção preditiva tradicional (que apenas detecta anomalias): ela utiliza **RAG (Retrieval-Augmented Generation)** integrado a modelos de linguagem (Google Gemini 2.0 Flash) e à base documental oficial da empresa para prescrever **exatamente quais ações de inspeção, ferramentas, procedimentos passo-a-passo e normas de segurança** devem ser adotadas.

### Principais Capacidades:
1. **Ingestão em Tempo Real:** Processamento de telemetria de sensores de vibração triaxial e temperatura (JSON).
2. **Classificação Multiclasse de Falhas (ML):** Diagnóstico automático do padrão de defeito via *Random Forest Classifier* com 84.2% de acurácia global e 85.4% F1-Macro.
3. **Motor de Busca por Similaridade Histórica:** Identificação de registros passados com comportamento idêntico (via *Nearest Neighbors* e similaridade de cosseno), apresentando frequência, distribuição temporal e contexto operacional (RPM e temperatura).
4. **Tratamento Documental Avançado (OCR + Chunking):** Extração de manuais digitais e escaneados (OCR via *RapidOCR ONNX* para Doc1.pdf) e indexação vetorial no *ChromaDB*.
5. **RAG Prescritivo com Guardrails Estritos:** Geração de prescrições técnicas baseadas estritamente nos manuais. Quando um defeito **não possui documentação**, o sistema alerta o usuário e sugere o cadastro de um novo procedimento.
6. **Dashboard Interativo & Chat Especialista:** Interface em *Streamlit* com gráficos Plotly em tempo real, chat conversacional com o técnico e área de upload de novos manuais.
7. **API REST Industrial:** Endpoints *FastAPI* documentados via OpenAPI/Swagger para integração com SCADA, MES e CLPs industriais.

---

## 🏗️ 2. Arquitetura da Solução e Implantação Industrial

```
                           ┌──────────────────────────────────────────────┐
                           │      SENSORES DE VIBRAÇÃO / CLP / SCADA      │
                           └──────────────────────┬───────────────────────┘
                                                  │ Telemetria JSON
                                                  ▼
                           ┌──────────────────────────────────────────────┐
                           │            FASTAPI GATEWAY REST              │
                           └──────┬───────────────────────────────┬───────┘
                                  │                               │
            ┌─────────────────────▼───────────────┐               │
            │   PIPELINE DE MACHINE LEARNING      │               │
            │  1. Feature Engineering (Ratios)    │               │
            │  2. StandardScaler                  │               │
            │  3. Random Forest Classifier        │               │
            │  4. NearestNeighbors Cosine Search  │               │
            └─────────────┬───────────────────────┘               │
                          │                                       │
            ┌─────────────▼───────────────────────┐               │
            │      HISTÓRICO & VECTOR STORE       │               │
            │  • SQLite (166k Leituras)           │               │
            │  • ChromaDB (Manuais Doc1 a Doc6)   │               │
            └─────────────┬───────────────────────┘               │
                          │                                       │
            ┌─────────────▼───────────────────────┐               │
            │   MOTOR RAG & GUARDRAILS (GEMINI)   │               │
            │  • Checagem de documento            │               │
            │  • Síntese Prescritiva Técnica      │               │
            │  • Guardrail: Falha sem doc         │               │
            └─────────────┬───────────────────────┘               │
                          │                                       │
                          ▼                                       ▼
            ┌───────────────────────────────────┐   ┌───────────────────────────┐
            │       STREAMLIT DASHBOARD         │   │   SISTEMAS EXTERNOS /     │
            │   • KPIs e Gráficos Plotly        │   │   INTEGRAÇÃO SCADA/MES    │
            │   • Análise de Novo Evento        │   │                           │
            │   • Chat Técnico com IA           │   │                           │
            │   • Gestão de Manuais PDF         │   │                           │
            └───────────────────────────────────┘   └───────────────────────────┘
```

### Arquitetura de Hardware & Limites Operacionais:
- **Estação de Trabalho Local:** Otimizado para rodar em estações com até **32 GB RAM e GPU de 16 GB** (utiliza menos de 1 GB de RAM e zero VRAM local devido à offload de embeddings e LLM para a API do Google Gemini).
- **Persistência:** SQLite local com índices otimizados e ChromaDB embutido sem necessidade de servidores externos pesados.

---

## 📊 3. Análise dos Dados e Mapeamento de Falhas

### Dataset: `banner.csv` (166.796 registros, 26 colunas)
- **Estados Operacionais Normais (sem defeito):** `normal`, `baseline`, `teste`, `acelerando`, `motor_desligado`, etc.
- **Falhas de Equipamento:** 130 variações rotuladas por operadores.

### Matriz de Cobertura Documental:
| Categoria de Defeito | Rótulos Brutos Mapeados | Documento Técnico Associado | Status RAG |
|---|---|---|---|
| **Rolamento (Pista Externa)** | `rolamento_outer*`, `new_rolamento_outer*` | **Doc1.pdf** (Páginas 2, 9, 13) | ✅ Coberto (OCR) |
| **Rolamento (Pista Interna)** | `rolamento_inner*`, `new_rolamento_inner*` | **Doc1.pdf** (Páginas 2, 10, 14) | ✅ Coberto (OCR) |
| **Rolamento (Elementos Rolantes)** | `rolamento_ball*`, `new_rolamento_ball*` | **Doc1.pdf** (Páginas 3, 10, 14) | ✅ Coberto (OCR) |
| **Rolamento (Gaiola / Combinação)**| `rolamento_comb*`, `new_rolamento_comb*` | **Doc1.pdf** (Páginas 3, 11, 14) | ✅ Coberto (OCR) |
| **Desalinhamento** | `desalinhado*`, `new_desalinhado*` | **Doc2.pdf** (Procedimento Completo) | ✅ Coberto |
| **Desbalanceamento** | `desbalanceado*`, `new_desbalanceado*` | **Doc3.pdf** (Procedimento Completo) | ✅ Coberto |
| **Correias de Transmissão** | `correia*`, `correia_2` | **Doc4.pdf** (Diagnóstico e Tensão) | ✅ Coberto |
| **Polias** | `polia*`, `polia_2` | **Doc5.pdf** (Alinhamento e Desgaste) | ✅ Coberto |
| **Rotor Inclinado (Cocked Rotor)**| `cocked_rotor*`, `new_cocked*` | **Doc6.pdf** (Batimento e Correção) | ✅ Coberto |
| **Rotor Excêntrico** | `eccentric_rotor*`, `new_eccentric*` | *(Nenhum documento inicial)* | ⚠️ **Guardrail Acionado** |
| **Ventoinha** | `ventoinha*`, `ventoinha_2` | *(Nenhum documento inicial)* | ⚠️ **Guardrail Acionado** |
| **Falta de Fase** | `new_falta_fase*` | *(Nenhum documento inicial)* | ⚠️ **Guardrail Acionado** |

---

## ⚡ 4. Como Executar a Aplicação

### 1. Clonar o repositório e criar ambiente virtual:
```bash
git clone <url-do-repositorio>
cd prescriptive_maintenance
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar dependências:
```bash
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente:
Copie o template `.env.example` para `.env` e insira sua chave do Google Gemini:
```env
GEMINI_API_KEY=sua_chave_do_google_ai_studio_aqui
PORT=8501
API_PORT=8000
```

### 4. Executar Pipeline de Treinamento e Ingestão:
```bash
python train_pipeline.py
```
*(Executa a criação do banco SQLite, pré-processamento de 166k dados, treinamento do classificador ML, OCR de Doc1.pdf e indexação vetorial no ChromaDB em ~50s).*

### 5. Iniciar a Aplicação Streamlit (Interface Visual):
```bash
streamlit run app.py
```
Acesse no navegador: **`http://localhost:8501`**

### 6. (Opcional) Iniciar a API REST FastAPI:
```bash
uvicorn api:app --reload --port 8000
```
Documentação interativa Swagger: **`http://localhost:8000/docs`**

---

## 🧪 5. Execução dos Testes Automatizados

O projeto conta com uma suíte de testes unitários e de integração cobrindo processamento de dados, modelo ML, busca por similaridade, RAG guardrails e endpoints da API:

```bash
python -m pytest tests/ -v
```

---

## 🚀 6. Deploy em Produção (Render / Docker)

### Opção A: Deploy no Render
O projeto já possui os arquivos [`Procfile`](file:///c:/Users/Augusto/Desktop/UTFPR/0%20-%20Emprego/0-Dev-fiesc-Antigravity-Claude/Procfile) e [`render.yaml`](file:///c:/Users/Augusto/Desktop/UTFPR/0%20-%20Emprego/0-Dev-fiesc-Antigravity-Claude/render.yaml) configurados.
1. Crie um novo **Web Service** no [Render Dashboard](https://render.com).
2. Conecte o repositório GitHub.
3. Configure a variável de ambiente `GEMINI_API_KEY`.
4. O build command executará automaticamente a instalação e inicialização dos modelos.

### Opção B: Deploy via Docker
```bash
docker build -t prescriptive-maintenance-ai .
docker run -p 8501:8501 -e GEMINI_API_KEY="sua_chave" prescriptive-maintenance-ai
```

---

## 👥 Estrutura do Repositório

```
├── config.py                 # Configurações globais e diretórios
├── requirements.txt          # Dependências Python
├── Procfile                  # Comando de inicialização para o Render
├── render.yaml               # Configuração de Infraestrutura como Código
├── Dockerfile                # Configuração de container Docker
├── .env.example              # Template de variáveis de ambiente
├── README.md                 # Documentação técnica completa
├── train_pipeline.py         # Pipeline completo de treino, OCR e ingestão
├── app.py                    # Aplicação Streamlit (Dashboard + Chat)
├── api.py                    # API REST FastAPI (Endpoints industriais)
├── src/
│   ├── constants.py          # Mapeamento de categorias e documentos
│   ├── data_processing.py    # Limpeza, engenharia de atributos e scaler
│   ├── classifier.py         # Classificador Random Forest (Treino e Inferência)
│   ├── similarity.py         # Busca por similaridade histórica (KNN Cosine)
│   ├── document_processor.py # PyMuPDF + RapidOCR + ChromaDB Vector Store
│   ├── rag_engine.py         # Motor Prescritivo RAG (Gemini + Guardrails)
│   └── database.py           # Gerenciamento SQLite, KPIs e logs
└── tests/
    ├── test_data_processing.py
    ├── test_classifier.py
    ├── test_similarity.py
    ├── test_rag_engine.py
    └── test_api.py
```
