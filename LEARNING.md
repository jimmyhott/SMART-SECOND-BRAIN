# Smart Second Brain — Comprehensive Learning Guide

This guide is written as a fast on-ramp for engineers who need to speak credibly about the Smart Second Brain project in design reviews and implementation discussions. Read end-to-end to internalize the architecture, key technologies, and practical lessons learned.

---

## 1. Elevator Pitch
- **Problem**: Knowledge workers accumulate research notes, PDFs, and conversations that are difficult to search and reuse.
- **Solution**: Smart Second Brain ingests raw documents, converts them into richly annotated vector embeddings, and exposes an AI assistant that answers questions with citations, supports human review, and stores validated knowledge for reuse.
- **Stack Highlights**: FastAPI, LangGraph, Azure/OpenAI embeddings & LLMs, ChromaDB, Streamlit, SQLite checkpoints, hybrid semantic + keyword retrieval, JSON-structured prompting, optional automatic metadata enrichment, and an extensive PyTest suite.

---

## 2. End-to-End Architecture
```
Streamlit UI → FastAPI (REST) → LangGraph workflows → ChromaDB (vectors)
                                      ↘ SQLite checkpoints (conversation + state)
Azure/OpenAI (embeddings + LLM) ────────────────────────────────────────────────┘
```
- **Streamlit Frontend**: Document ingestion, query interface, retrieval options, and HITL feedback.
- **FastAPI Backend**: Request validation, collection routing, async LangGraph execution, health & admin endpoints.
- **LangGraph**: Orchestrates ingestion and query workflows, tracks KnowledgeState, and manages human review interrupts.
  - Auto preprocessing node enriches documents with inferred categories/keywords when explicit metadata is missing.
- **ChromaDB**: Persistent vector store with collection isolation, metadata filtering, and hybrid retrieval hooks.
- **SQLite**: LangGraph checkpointing (conversation memory, workflow resumption) with per-test isolation in CI.
- **Azure/OpenAI**: Text embeddings (`text-embedding-3-small`) and chat completions (e.g., `gpt-4o`) with JSON-formatted prompts.
  - Auto metadata enrichment flow runs before chunking when callers opt in.

---

## 3. Core Technologies & Implementation Details
### 3.1 LangGraph (Workflow Orchestration)
- **KnowledgeState** (Pydantic model) tracks raw docs, chunks, embeddings, metadata, retrieval logs, citation maps, and HITL flags.
- **Nodes**:
  - `chunk_doc_node`: Splits documents (RecursiveCharacterTextSplitter), assigns deterministic `doc_id` & `chunk_id`, stores metadata (source, knowledge type, char offsets).
  - `preprocess_node`: Optional step that validates user-supplied metadata or infers keywords/categories using lightweight text analysis.
  - `embed_node`: Calls embeddings (Azure/OpenAI), stores vectors alongside metadata.
  - `store_node`: Transforms chunks + metadata into LangChain `Document` objects, assigns stable vector IDs (`doc_id::chunk_n`), writes them to ChromaDB, and forwards the same documents to `SmartDocumentRetriever` so dense and BM25 retrievers stay in sync.
  - `retriever_node`: Accepts retrieval options (k, min_score, use_hybrid, rerank_top_k, metadata filters), logs results with provenance, and populates citation map.
  - `answer_gen_node`: Loads `agentic/prompts/answer_prompt.txt`, ensures the prompt contains the `citations` variable, invokes the LLM, and falls back to a structured “IDK” JSON payload on errors.
  - `human_review_node` & `validated_store_node`: Support human-in-the-loop review, approvals, edited answers, and optional storage of validated knowledge.
- **Checkpointing**: Default path `data/checkpoints.sqlite`; tests override with temp files to avoid corruption.

### 3.2 FastAPI Backend
- **Entry Point**: `api/main.py` wires routers, CORS, logging, and lifecycle events.
- **Routes** (`api/routes/v1/graph_api.py`):
  - `POST /graph/ingest`: Builds/loads a collection-scoped `MasterGraphBuilder`, invokes ingestion workflow, returns thread ID and status.
  - `POST /graph/query`: Accepts query text, optional thread ID, metadata filters, retrieval options; executes LangGraph query workflow.
  - `POST /graph/feedback`: Resumes interrupted workflows with human feedback (approved/rejected/edited).
  - `DELETE /graph/clear-vector-db`: Maintenance endpoint with optional backup.
  - `GET /graph/health`: Aggregates backend, vector store, and LLM readiness.
- **Settings**: Pydantic `Config` (env-driven), `.env` templated.

### 3.3 SmartDocumentRetriever (Semantic + Keyword Search)
- **Vector Search**: Uses Chroma’s dense similarity; each document chunk includes metadata for filtering.
- **BM25 Retriever**: Instantiated lazily; supports sparse keyword search.
  - Requires installing `rank_bm25`; tests log a warning when the dependency is absent, but dense retrieval still works.
- **Hybrid Ensemble**: Weighted combination (`0.7` semantic, `0.3` BM25) via LangChain’s `EnsembleRetriever`.
- **Reranking**: Lightweight heuristic (content length, metadata richness, recency) to reduce near-duplicate top-k results.
- **Configurable Retrieval Options**: Frontend exposes `k`, `min_score`, `use_hybrid`, `rerank_top_k`, metadata filters (sources, categories, knowledge types).
- **Observability**: Retrieval logs capture filters, options, scores, and metadata for each run—useful in debugging and stakeholder conversations.
  - Metadata sanitization ensures Chroma receives scalar-friendly values even when auto preprocessing produces lists.

### 3.4 Vector Storage & Retrieval Deep Dive
1. **Chunk Normalization**
   - Each chunk inherits `doc_id`, `chunk_id`, categories, knowledge type, source, timestamp, and character offsets.
   - Deterministic IDs make deletions/replacements and citation lookups trivial.
2. **Store Node Mechanics**
   - Converts chunk/metadata pairs into LangChain `Document` objects.
   - Persists to Chroma via `add_documents(documents, ids=ids)` ensuring we control vector IDs.
   - Immediately mirrors the same documents into `SmartDocumentRetriever` so the dense and BM25 retrievers are always aligned.
3. **ChromaDB Configuration**
   - Per-collection persistence directory (`./chroma_db/<collection>`), enabling multi-tenant knowledge bases.
   - Metadata is stored alongside embeddings, enabling filters (`source`, `categories`, `knowledge_type`).
4. **Hybrid Retrieval Pipeline**
   - Dense retriever: semantic similarity with configurable `k` and `min_score`.
   - BM25 retriever: keyword fallback for rare terms.
   - Ensemble retriever: weighted merge (70% dense / 30% sparse).
   - Optional rerank: boosts longer, metadata-rich, or recent chunks before final selection.
5. **Metadata-Driven Filtering**
   - API and UI support filter dictionaries (`{"sources": [...], "categories": [...]}`) that flow to Chroma search filters.
   - Allows isolation by document provenance, topic, or knowledge lifecycle stage.
6. **Retrieval Logging & Citation Map**
   - Retrieval node logs the query, options, filters, and a summary of each hit (doc_id, chunk_id, source, score).
   - Citation map feeds the prompt and surfaces in the API response so the frontend can render inline references.

### 3.5 Prompt Engineering & JSON Responses
- **Prompt Template**: `agentic/prompts/answer_prompt.txt` lists context, conversation, and available citations, instructs the LLM to return JSON with `answer`, `citations[]`, `is_idk`, and `confidence`.
- **Structured Output Handling**: `answer_gen_node` inspects prompt input variables, injects citation map, and on failure emits a fallback JSON (IDK) with an embedded error message. This ensures the frontend never sees raw prompt errors.

### 3.6 Streamlit Frontend
- **State Management**: Uses `st.session_state` for metadata filters, retrieval options, and chat history.
- **UI Modules**:
  - Sidebar: Collection selection, retrieval filters (sources, categories, knowledge types), retrieval tunables (k, min score, hybrid, rerank).
  - Main Panel: Document upload, ingestion forms, chat interface showing answers with citations, feedback buttons.
  - Health Dashboard: Polls backend health endpoint.
- **API Integration**: Calls FastAPI routes, handles JSON responses, displays the structured answer and citations.

### 3.7 Testing Strategy
- **Unit & Service Tests**: `tests/test_services/test_master_graph_builder.py`, `tests/test_services/test_document_retriever.py` mock LLMs, vector stores, and retrievers.
- **API Tests**: `tests/test_api/test_graph_api.py` cover ingestion, query, feedback, clear DB, and health endpoints.
- **Integration Considerations**: Real integration tests require Azure/OpenAI keys; vector DB and checkpoint paths are isolated per test via temp directories, and each test writer injects a temp SQLite file into the store node to prevent leftover state.
- **Fixtures**: Provide mock LLMs returning deterministic JSON, retrieval results with `.document` + `.score`, and per-test SQLite `SqliteSaver` checkpoints.
- **Challenges Solved**: Avoided SQLite corruption by using temp checkpoint DBs, adjusted tests to accept fallback `status="error"` responses when LLM generation fails, and synchronized prompt mocks with new `citations` variable.

---

## 4. Detailed Workflows & Data Flows
### 4.1 Document Ingestion ("ETL") Workflow
- **API Request** (`/graph/ingest`): Receives raw text or PDF metadata, auto_preprocess flag, and optional collection name.
- **Auto Metadata Preprocessing**: If `auto_preprocess` is true, `preprocess_node` checks for existing metadata, invokes spaCy (`en_core_web_sm`) to lemmatize the document, filters out scikit-learn English stopwords, and infers keywords/categories. Results are merged with caller metadata. If spaCy isn't installed, a regex-based tokenizer provides a fallback so ingestion isn't blocked. When auto preprocessing is off and no metadata is supplied, ingestion short-circuits with `status="skipped_no_metadata"`.
- **Chunking**: `RecursiveCharacterTextSplitter` creates 500-character chunks with 50-character overlap. Each chunk inherits deterministic `doc_id`/`chunk_id`, character offsets, categories, keywords, knowledge type, ingestion timestamp, and source, captured in `state.chunk_metadata` for later filtering and citation mapping.
- **Embedding**: Azure/OpenAI embeddings (`text-embedding-3-small` by default) transform chunks into vectors. Failures fallback to placeholder vectors so the pipeline remains testable.
- **Vector Storage**: `store_node` serializes metadata into scalar-friendly strings, persists chunks to ChromaDB via deterministic IDs, and mirrors the same documents into `SmartDocumentRetriever` so dense and BM25 modes stay synchronized.
- **Completion**: Workflow logs chunk counts, embedding status, and returns final `KnowledgeState.status` (`stored`, `validated`, `error`, etc.). Human-in-the-loop nodes can pause/resume the flow for approvals.

### 4.2 Query Workflow (RAG Loop)
1. **API Request** (`/graph/query`): Accepts query, thread ID, metadata filters, retrieval options.
2. **Retrieval**:
   - Hybrid retriever (semantic + BM25) fetches top candidates.
   - Optional reranking adjusts scores.
   - Retrieval log records filters, scores, and metadata.
3. **Answer Generation**:
   - Builds conversation history and context snippets (with headers `[Source N]`).
   - Loads prompt, injects citation map, and invokes LLM.
   - Falls back to IDK JSON if LLM fails; logs the outcome.
4. **Human Review** (optional): If `knowledge_type` implies reusable/verified and no feedback provided, LangGraph raises an interrupt; user approves/edits via `/graph/feedback`.
5. **Validated Storage**: Approved answers can be stored for future retrieval (knowledge base growth).
6. **Response**: Returns structured JSON with `generated_answer`, `retrieved_docs`, `citation_map`, `status`, `logs`, and `thread_id` for continuity.

### 4.3 Human Feedback Loop
- **Interrupt**: Workflow pauses when review required; returns instruction to client.
- **Resume**: `/graph/feedback` with `thread_id`, `feedback` (`approved|rejected|edited`), optional `edited_answer`, and `knowledge_type` resumes the graph.
- **Outcome**: Final state includes `final_answer`, `human_feedback`, and updated status (`validated`, `validated_no_store`).

---

## 5. Data & Storage Model
| Component         | Technology | Purpose                                              | Notes                                               |
|-------------------|------------|------------------------------------------------------|-----------------------------------------------------|
| Vector Store      | ChromaDB   | Dense embeddings, metadata filters, hybrid search    | Collection per knowledge base; disk persistence     |
| Workflow State    | SQLite     | LangGraph checkpoints, conversation threads          | Per-test temp DB in CI; default `data/checkpoints.sqlite` |
| Metadata          | Pydantic   | `KnowledgeState` models (messages, chunks, citations) | Strong typing, validation, JSON serialization       |
| Logs & Monitoring | Python logging | Centralized logging (JSON-friendly format)       | `shared/utils/logging_config.py` sets up handlers   |

---

## 6. Deployment & Operations
- **Startup**: `start_system.py` boots backend and frontend, waits for health checks, and monitors processes.
- **Shutdown**: `stop_system.py` gracefully terminates all components.
- **Environment**: `.env` contains OpenAI/Azure credentials and optional server config. Default ports: backend 8000, frontend 5173.
- **Clear Vector DB**: Script and API endpoint support backups and selective deletion.
- **Observability**: Health endpoint aggregates component states; logs capture retrieval diagnostics, prompt inputs, and fallback reasons.

---

## 7. Key Lessons & Talking Points
1. **Hybrid Retrieval Matters**: Pure vector search struggled with rare keywords; combining BM25 improved recall without losing semantic context.
2. **Metadata-Rich Chunks Enable Filtering & Citations**: Storing source, categories, knowledge type, and offsets supports traceable answers and UI filters.
3. **Structured Prompts Reduce Frontend Fragility**: Forcing JSON with citation arrays simplified parsing and made failure modes explicit (fallback JSON with error field).
4. **LangGraph Checkpoint Isolation**: Sharing a single SQLite file corrupted tests; per-test temp DBs solved concurrency issues and mirror production isolation.
5. **Human-in-the-Loop via Interrupts**: LangGraph interrupts provided a clean way to pause workflows for human review, resume via API, and maintain conversation state.
6. **Streamlit as a Rapid Control Plane**: Session state and sidebar controls made it easy to expose retrieval knobs (k, min_score, rerank) for experimentation.
7. **Testing Strategy**: Extensive mocking ensured deterministic tests; we learned to align prompt mocks with new variables (`citations`) and accept fallback statuses when LLM calls fail.

---

## 8. Quick Reference (Technologies & Responsibilities)
| Layer       | Tech / Library             | Responsibility                                                |
|-------------|----------------------------|----------------------------------------------------------------|
| UI          | Streamlit                  | Document ingestion, chat UI, retrieval controls, feedback     |
| API         | FastAPI + Pydantic         | Request validation, routing, state management, health checks  |
| Workflow    | LangGraph                  | Ingestion & query pipelines, HITL interrupts, state persistence |
| Retrieval   | ChromaDB + BM25 Ensemble   | Hybrid semantic/keyword search, metadata filtering, reranking |
| AI Models   | Azure/OpenAI (Embeddings & LLM) | Chunk embeddings, citation-oriented answer generation   |
| Storage     | SQLite                     | LangGraph checkpoints (conversation + workflow state)         |
| Logging     | Python logging             | Centralized structured logging, retrieval diagnostics         |
| Testing     | PyTest                     | Unit, service, API, integration tests with isolated fixtures  |

---

## 9. Further Reading & Practice
- **LangGraph Docs**: Dive into state management, interrupts, and workflow design patterns.
- **ChromaDB & Hybrid Retrieval**: Experiment with metadata filters, reranking strategies, and potential MMR integration.
- **Prompt Engineering**: Explore few-shot prompts with citation formatting, JSON schema enforcement, and error handling strategies.
- **Ops**: Investigate containerizing FastAPI + Streamlit, adding metrics (Prometheus), and enabling distributed vector stores (Pinecone/Weaviate).

---

Armed with this guide, you should be ready to explain the Smart Second Brain architecture, justify design decisions, and discuss concrete implementation details in any technical deep dive. Good luck!
