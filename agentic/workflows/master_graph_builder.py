"""
Master Graph Builder for Smart Second Brain LangGraph Workflows.

This module implements the core workflow orchestration system for the Smart Second Brain
project. It defines a comprehensive LangGraph workflow that handles both document
ingestion and knowledge querying through a series of interconnected nodes.

Key Workflows:
1. Document Ingestion: raw_document -> chunking -> embedding -> storage
2. Knowledge Query: user_input -> retrieval -> generation -> validation -> storage

Core Components:
- StateGraph: LangGraph workflow definition with conditional routing
- Node Methods: Individual processing steps for each workflow stage
- Checkpointing: Memory-based state persistence for conversation continuity
- Error Handling: Comprehensive error handling with fallback mechanisms

Architecture:
- Router Node: Determines workflow path based on query_type
- Ingestion Branch: Document processing and storage pipeline
- Query Branch: Knowledge retrieval and answer generation pipeline
- Human Review: Human-in-the-loop validation and feedback
- Validated Storage: Persistent storage of approved answers

Dependencies:
- LangGraph: Workflow orchestration and state management
- LangChain: Text processing, embeddings, and LLM integration
- ChromaDB: Vector database for document storage and retrieval
- Azure OpenAI: Embedding generation and language model interactions

Author: Smart Second Brain Team
Version: 0.1.0
"""

from langgraph.graph import StateGraph, END
try:
    # LangGraph interrupt for human-in-the-loop pauses
    from langgraph.types import interrupt  # type: ignore
except Exception:  # Fallback if interrupt is unavailable
    interrupt = None  # type: ignore
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os
import uuid
import datetime
import json
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
try:
    import spacy
except ImportError:  # pragma: no cover - spaCy optional
    spacy = None
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from pathlib import Path
from ..core.knowledge_state import KnowledgeState
from .document_retriever import SmartDocumentRetriever

# Import centralized logging
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from shared.utils.logging_config import setup_agentic_logging

# Set up agentic logging
logger = setup_agentic_logging()


SKLEARN_STOPWORDS = set(ENGLISH_STOP_WORDS)
_SPACY_NLP = None
_SPACY_LOAD_FAILED = False


def get_spacy_nlp():
    global _SPACY_NLP, _SPACY_LOAD_FAILED
    if _SPACY_LOAD_FAILED or spacy is None:
        return None
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception as exc:  # pragma: no cover - dependent on runtime env
            logger.warning("spaCy model load failed (%s); falling back to regex tokenizer", exc)
            _SPACY_LOAD_FAILED = True
            return None
    return _SPACY_NLP


class MasterGraphBuilder:
    """
    Main workflow orchestrator for Smart Second Brain knowledge processing.
    
    This class builds and manages the LangGraph workflow that handles both
    document ingestion and knowledge querying. It encapsulates all node
    definitions, workflow routing logic, and state management.
    
    The workflow supports two main paths:
    1. Document Ingestion: Processes new documents into the knowledge base
    2. Knowledge Query: Retrieves and generates answers from stored knowledge
    
    Key Features:
    - Conditional workflow routing based on query type
    - Comprehensive error handling and fallback mechanisms
    - Human-in-the-loop review and validation
    - Persistent state management with checkpointing
    - Flexible integration with various LLM and embedding providers
    
    Attributes:
        llm: Language model for text generation and reasoning
        embedding_model: Model for generating text embeddings
        retriever: Document retriever for similarity search
        vectorstore: Vector database for document storage
        chromadb_dir: Directory for ChromaDB persistence
        checkpointer: Memory-based state persistence system
    """

    def __init__(self, llm=None, embedding_model=None, retriever=None, vectorstore=None, chromadb_dir=None, collection_name=None):
        """
        Initialize the MasterGraphBuilder with required dependencies.
        
        Args:
            llm: Language model client (e.g., ChatOpenAI, AzureChatOpenAI)
            embedding_model: Text embedding model (e.g., OpenAIEmbeddings, AzureOpenAIEmbeddings)
            retriever: Document retriever for similarity search (optional)
            vectorstore: Vector database client (optional, will be created if not provided)
            chromadb_dir: ChromaDB persistence directory (defaults to "./chroma_db")
            collection_name: ChromaDB collection name (defaults to "smart_second_brain")
            
        Note:
            At minimum, either llm or embedding_model should be provided for
            basic functionality. For full RAG capabilities, both are required.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.chromadb_dir = chromadb_dir or "./chroma_db"
        self.collection_name = collection_name or "smart_second_brain"
        
        # Initialize SqliteSaver checkpointer (no external service required)
        # Stores checkpoints in ./data/checkpoints.sqlite
        os.makedirs("data", exist_ok=True)
        sqlite_conn = sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(sqlite_conn)

        # Ensure vectorstore and retriever are initialized consistently
        if not self.vectorstore and self.embedding_model:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.chromadb_dir,
            )

        if not self.retriever and self.vectorstore:
            self.retriever = SmartDocumentRetriever(
                vectorstore=self.vectorstore,
                embedding_model=self.embedding_model,
                collection_name=self.collection_name,
                chromadb_dir=self.chromadb_dir,
            )

    # =============================================================================
    # WORKFLOW NODE METHODS
    # =============================================================================

    def input_router(self, state: KnowledgeState):
        """
        Route incoming requests to appropriate workflow branches.
        
        This node acts as the entry point for all workflows and determines
        the processing path based on the query_type field in the state.
        
        Args:
            state: KnowledgeState object containing workflow parameters
            
        Returns:
            KnowledgeState: Updated state object (unchanged for valid routes)
            
        Workflow Routes:
            - "ingest": Document ingestion workflow
            - "query": Knowledge query workflow
            - Unknown: Error state with logging
        """
        if state.query_type == "ingest":
            return state
        elif state.query_type == "query":
            return state
        else:
            state.status = "error"
            state.logs = (state.logs or []) + ["Unknown query_type"]
            return state

    def preprocess_node(self, state: KnowledgeState):
        """
        Perform optional automated metadata extraction prior to chunking.

        When auto_preprocess is enabled, this node analyses the raw document to
        infer keywords and categories. When disabled, it ensures the caller has
        supplied sufficient metadata; otherwise the ingestion run is skipped.
        """
        if state.query_type != "ingest":
            return state

        if not state.raw_document:
            state.logs = (state.logs or []) + ["⚠️ No raw document present for preprocessing"]
            return state

        if not state.auto_preprocess:
            has_user_metadata = bool((state.categories or []) or (state.metadata or {}))
            if not has_user_metadata:
                state.status = "skipped_no_metadata"
                state.logs = (state.logs or []) + [
                    "⚠️ Ingestion skipped: auto_preprocess disabled and no metadata provided"
                ]
            return state

        # --- Tokenize and normalize text ---
        filtered_tokens: list[str] = []
        nlp = get_spacy_nlp()
        if nlp is not None:
            doc = nlp(state.raw_document)
            for token in doc:
                lemma = token.lemma_.lower().strip()
                if (
                    lemma
                    and lemma.isalpha()
                    and len(lemma) >= 4
                    and lemma not in SKLEARN_STOPWORDS
                ):
                    filtered_tokens.append(lemma)
        else:
            text = state.raw_document.lower()
            tokens = re.findall(r"[a-zA-Z]{4,}", text)
            filtered_tokens = [token for token in tokens if token not in SKLEARN_STOPWORDS]

        if not filtered_tokens:
            state.logs = (state.logs or []) + [
                "ℹ️ Auto preprocessing found no candidate keywords"
            ]
            return state

        counter = Counter(filtered_tokens)
        keywords = [word for word, _ in counter.most_common(10)]
        categories = keywords[:5]

        state.auto_generated_keywords = keywords
        state.auto_generated_categories = categories

        # Merge inferred categories with any caller-supplied ones while preserving order
        existing_categories = state.categories or []
        merged_categories = list(dict.fromkeys(existing_categories + categories))
        state.categories = merged_categories

        state.metadata = state.metadata or {}
        if keywords:
            state.metadata.setdefault("keywords", ", ".join(keywords))
            state.metadata.setdefault("auto_generated_keywords", ", ".join(keywords))
        if merged_categories:
            state.metadata.setdefault("categories", ", ".join(merged_categories))
            state.metadata.setdefault("auto_generated_categories", ", ".join(categories))
        state.metadata.setdefault("auto_preprocess", True)

        state.logs = (state.logs or []) + [
            f"🔍 Auto preprocessing inferred keywords={keywords[:5]} categories={categories[:3]}"
        ]

        return state

    def chunk_doc_node(self, state: KnowledgeState):
        """
        Split raw document text into manageable chunks for embedding.
        
        This node processes the raw document text and splits it into
        overlapping chunks that are optimal for embedding generation.
        Uses RecursiveCharacterTextSplitter for intelligent text segmentation.
        
        Args:
            state: KnowledgeState object containing raw_document
            
        Returns:
            KnowledgeState: Updated state with chunks array
            
        Processing:
            - Splits text into 500-character chunks with 50-character overlap
            - Maintains semantic coherence within chunks
            - Logs chunking statistics for monitoring
            - Handles missing document gracefully
        """
        if getattr(state, "status", None) == "skipped_no_metadata":
            return state

        if state.raw_document:
            doc_id = state.doc_id or f"doc_{uuid.uuid4()}"
            state.doc_id = doc_id
            state.ingested_at = state.ingested_at or datetime.datetime.utcnow().isoformat()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )
            chunks = splitter.split_text(state.raw_document)
            state.chunks = chunks
            state.chunk_metadata = []

            for index, chunk in enumerate(chunks):
                metadata = {
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "chunk_id": f"{doc_id}::chunk_{index}",
                    "source": state.source or "unknown",
                    "categories": state.categories or [],
                    "keywords": (state.auto_generated_keywords or []) or (
                        state.metadata.get("keywords") if state.metadata else []
                    ),
                    "ingested_at": state.ingested_at,
                    "knowledge_type": state.knowledge_type or "conversational",
                    "char_start": state.raw_document.find(chunk),
                }
                metadata["char_end"] = metadata["char_start"] + len(chunk) if metadata["char_start"] >= 0 else None
                state.chunk_metadata.append(metadata)

            state.logs = (state.logs or []) + [
                f"Chunked document into {len(state.chunks)} chunks with doc_id={doc_id}"
            ]
        else:
            state.logs = (state.logs or []) + ["No raw_document found for chunking"]

        return state

    def embed_node(self, state: KnowledgeState):
        """
        Generate vector embeddings for text chunks.
        
        This node converts text chunks into high-dimensional vector
        representations that capture semantic meaning for similarity search.
        Includes fallback mechanisms for error handling and missing models.
        
        Args:
            state: KnowledgeState object containing chunks array
            
        Returns:
            KnowledgeState: Updated state with embeddings array
            
        Processing:
            - Generates embeddings using the configured embedding model
            - Falls back to placeholder embeddings if model fails
            - Logs embedding generation progress and errors
            - Handles missing chunks gracefully
        """
        if getattr(state, "status", None) == "skipped_no_metadata":
            state.logs = (state.logs or []) + [
                "ℹ️ Skipping embedding because ingestion was skipped"
            ]
            return state

        # Use embedding model if available, otherwise fallback to placeholder
        if state.chunks:
            if self.embedding_model:
                try:
                    logger.info(f"🔤 Generating embeddings for {len(state.chunks)} chunks")
                    state.embeddings = self.embedding_model.embed_documents(state.chunks)
                    logger.info(f"✅ Generated {len(state.embeddings)} embeddings")
                except Exception as e:
                    logger.error(f"❌ Embedding generation failed: {e}")
                    # Fallback to placeholder embeddings for continued processing
                    state.embeddings = [[0.1, 0.2]] * len(state.chunks)
                    state.logs = (state.logs or []) + [f"Embedding error: {str(e)}"]
            else:
                logger.warning("⚠️ No embedding model provided, using placeholder embeddings")
                state.embeddings = [[0.1, 0.2]] * len(state.chunks)
        return state

    def store_node(self, state: KnowledgeState):
        """
        Store embeddings and chunks into ChromaDB vector database.
        
        This node persists the processed document chunks and their
        embeddings into the vector database for future retrieval.
        Includes comprehensive metadata for organization and traceability.
        
        Args:
            state: KnowledgeState object containing chunks, embeddings, and metadata
            
        Returns:
            KnowledgeState: Updated state with storage status
            
        Processing:
            - Creates ChromaDB collection if not exists
            - Stores text chunks with metadata (source, categories, chunk_id)
            - Handles metadata formatting for ChromaDB compatibility
            - Logs storage operations and errors
            - Updates state status upon completion
        """
        if getattr(state, "status", None) == "skipped_no_metadata":
            state.logs = (state.logs or []) + [
                "ℹ️ Skipping storage because ingestion was skipped"
            ]
            return state

        if state.embeddings and state.chunks:
            try:
                # Initialize vectorstore if not provided
                if not self.vectorstore:
                    self.vectorstore = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embedding_model,
                        persist_directory=self.chromadb_dir
                    )

                metadatas = []
                ids = []
                for idx, chunk in enumerate(state.chunks):
                    base_metadata = state.chunk_metadata[idx] if state.chunk_metadata and idx < len(state.chunk_metadata) else {}
                    metadata = {
                        "doc_id": base_metadata.get("doc_id", state.doc_id or "unknown"),
                        "chunk_index": base_metadata.get("chunk_index", idx),
                        "chunk_id": base_metadata.get("chunk_id", f"chunk_{idx}"),
                        "source": base_metadata.get("source", state.source or "unknown"),
                        "categories": base_metadata.get("categories", state.categories or []),
                        "ingested_at": base_metadata.get("ingested_at", state.ingested_at),
                        "knowledge_type": base_metadata.get("knowledge_type", state.knowledge_type or "conversational"),
                        "char_start": base_metadata.get("char_start"),
                        "char_end": base_metadata.get("char_end"),
                    }
                    metadata.update(state.metadata or {})

                    safe_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            safe_metadata[key] = value
                        elif isinstance(value, list):
                            safe_metadata[key] = ", ".join(str(v) for v in value)
                        else:
                            safe_metadata[key] = str(value)

                    metadatas.append(safe_metadata)
                    ids.append(metadata["chunk_id"])

                documents = [
                    Document(page_content=chunk, metadata=meta)
                    for chunk, meta in zip(state.chunks, metadatas)
                ]

                # Store chunks in vector store and update hybrid retriever if available
                if self.vectorstore:
                    if hasattr(self.vectorstore, "add_documents"):
                        self.vectorstore.add_documents(documents, ids=ids)
                    elif hasattr(self.vectorstore, "add_texts"):
                        self.vectorstore.add_texts([d.page_content for d in documents], metadatas=metadatas, ids=ids)

                if self.retriever:
                    self.retriever.add_documents(documents)

                # Note: Newer ChromaDB versions don't require explicit persist()
                # Data is automatically persisted when using persist_directory

                state.status = "stored"
                state.logs = (state.logs or []) + [
                    f"Stored {len(state.chunks)} chunks in ChromaDB with categories {state.categories or ['general']}"
                ]
            except Exception as e:
                state.status = "error"
                state.logs = (state.logs or []) + [f"Storing failed: {e}"]
        else:
            state.logs = (state.logs or []) + ["No embeddings/chunks to store"]

        return state

    def retriever_node(self, state: KnowledgeState):
        """
        Retrieve relevant documents from the vector database.
        
        This node performs similarity search to find the most relevant
        documents for a given user query. Supports multiple retrieval
        strategies with fallback mechanisms.
        
        Args:
            state: KnowledgeState object containing user_input
            
        Returns:
            KnowledgeState: Updated state with retrieved_docs array
            
        Retrieval Strategies:
            1. Dedicated retriever (if configured)
            2. Vectorstore similarity search (fallback)
            3. Empty results (if both unavailable)
            
        Processing:
            - Performs semantic search using user query
            - Retrieves top-k most relevant documents
            - Formats results for downstream processing
            - Handles retrieval errors gracefully
            - Logs retrieval operations and statistics
        """
        if not state.user_input:
            state.logs = (state.logs or []) + ["No user_input provided for retrieval"]
            state.retrieved_docs = []
            return state

        try:
            retrieval_options = {
                "query": state.user_input,
                "k": (state.retrieval_options or {}).get("k", 8),
                "filters": state.metadata_filters or {},
                "use_hybrid": (state.retrieval_options or {}).get("use_hybrid", True),
                "min_score": (state.retrieval_options or {}).get("min_score", 0.1),
                "rerank_top_k": (state.retrieval_options or {}).get("rerank_top_k", 12),
            }

            retrieval_log = {
                "query": state.user_input,
                "filters": retrieval_options["filters"],
                "use_hybrid": retrieval_options["use_hybrid"],
                "min_score": retrieval_options["min_score"],
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }

            if self.retriever:
                logger.info(f"🔍 Retrieving docs for query: {state.user_input}")
                if hasattr(self.retriever, "retrieve"):
                    results = self.retriever.retrieve(
                        query=retrieval_options["query"],
                        k=retrieval_options["k"],
                        filters=retrieval_options["filters"],
                        use_hybrid=retrieval_options["use_hybrid"],
                        min_score=retrieval_options["min_score"],
                    )
                elif hasattr(self.retriever, "get_relevant_documents"):
                    results = self.retriever.get_relevant_documents(retrieval_options["query"])
                else:
                    results = []

                # Optionally apply reranking if supported
                if (
                    retrieval_options["rerank_top_k"]
                    and hasattr(self.retriever, "retrieve_with_reranking")
                ):
                    results = self.retriever.retrieve_with_reranking(
                        query=retrieval_options["query"],
                        k=retrieval_options["k"],
                        rerank_top_k=retrieval_options["rerank_top_k"],
                    )

                formatted_results = []
                top_results = []
                for result in results:
                    doc = result.document if hasattr(result, "document") else result
                    score = result.score if hasattr(result, "score") else None
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    })
                    top_results.append({
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "doc_id": doc.metadata.get("doc_id"),
                        "source": doc.metadata.get("source"),
                        "score": score,
                    })
                state.retrieved_docs = formatted_results
                retrieval_log["results"] = top_results

            elif self.vectorstore:
                logger.info(f"🔍 Using vectorstore.similarity_search for query: {state.user_input}")
                results = self.vectorstore.similarity_search(
                    retrieval_options["query"],
                    k=retrieval_options["k"],
                    filter=retrieval_options["filters"] or None,
                )
                state.retrieved_docs = [
                    {"content": r.page_content, "metadata": r.metadata, "score": None}
                    for r in results
                ]
                retrieval_log["results"] = [
                    {
                        "chunk_id": r.metadata.get("chunk_id"),
                        "doc_id": r.metadata.get("doc_id"),
                        "source": r.metadata.get("source"),
                        "score": None,
                    }
                    for r in results
                ]
            else:
                state.retrieved_docs = []
                retrieval_log["results"] = []

            state.retrieval_log = (state.retrieval_log or []) + [retrieval_log]
            state.logs = (state.logs or []) + [f"Retrieved {len(state.retrieved_docs)} docs"]

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            state.retrieved_docs = []
            state.retrieval_log = (state.retrieval_log or []) + [{
                "query": state.user_input,
                "error": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }]
            state.logs = (state.logs or []) + [f"Retrieval error: {str(e)}"]

        return state

    def answer_gen_node(self, state: KnowledgeState):
        """
        Generate AI-powered answers using retrieved context and conversation history.
        
        This node implements the core RAG (Retrieval-Augmented Generation) step,
        combining retrieved documents with conversation context to generate
        grounded, relevant answers to user queries.
        
        Args:
            state: KnowledgeState object containing user_input, retrieved_docs, and messages
            
        Returns:
            KnowledgeState: Updated state with generated_answer and updated messages
            
        RAG Implementation:
            - Context: Retrieved document content
            - Conversation: User-assistant message history
            - Query: Current user question
            - Instructions: Clear guidelines for answer generation
            
        Processing:
            - Builds context from retrieved documents
            - Formats conversation history
            - Uses ChatPromptTemplate for structured prompting
            - Calls language model with full context
            - Updates conversation history
            - Logs generation success/failure
        """
        if not self.llm or not state.user_input:
            state.logs = (state.logs or []) + ["⚠️ No LLM or user query provided"]
            return state

        try:
            # --- Build retrieved context ---
            citation_map = []
            context_chunks = []
            for idx, doc in enumerate(state.retrieved_docs or []):
                citation_id = idx + 1
                metadata = doc.get("metadata", {})
                snippet_header = f"[Source {citation_id}] {metadata.get('source', 'unknown')}"
                context_chunks.append(f"{snippet_header}\n{doc['content']}")
                citation_entry = {
                    "citation_id": citation_id,
                    "chunk_id": metadata.get("chunk_id"),
                    "doc_id": metadata.get("doc_id"),
                    "source": metadata.get("source"),
                    "score": doc.get("score"),
                    "metadata": metadata,
                }
                citation_map.append(citation_entry)

            context = "\n\n".join(context_chunks) or "No relevant documents were retrieved."
            state.citation_map = citation_map

            # --- Format conversation history ---
            conversation = "\n".join(
                [f"{m['role']}: {m['content']}" for m in (state.messages or [])]
            )

            # --- Load externalized prompt template ---
            prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "answer_prompt.txt"
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm

            # --- Call language model with full context ---
            input_variables = {
                "conversation": conversation,
                "query": state.user_input,
                "context": context,
            }
            if hasattr(prompt, "input_variables") and "citations" in prompt.input_variables:
                formatted_citations = [
                    {
                        "id": entry.get("citation_id"),
                        "doc_id": entry.get("doc_id"),
                        "chunk_id": entry.get("chunk_id"),
                        "source": entry.get("source"),
                        "score": entry.get("score"),
                        "metadata": entry.get("metadata", {}),
                    }
                    for entry in citation_map
                ]
                input_variables["citations"] = json.dumps(formatted_citations, ensure_ascii=False)
            logger.debug(
                "Answer prompt inputs: required=%s provided=%s",
                getattr(prompt, "input_variables", []),
                list(input_variables.keys()),
            )

            # --- Store the raw LLM response for frontend parsing ---
            response = chain.invoke(input_variables)
            content = getattr(response, "content", response)
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            elif content is None:
                raise ValueError("LLM returned no content")
            state.generated_answer = str(content)
            state.is_idk_response = None  # Let frontend determine this
            state.logs = (state.logs or []) + ["✅ Generated LLM response for frontend parsing"]
            
            # Initialize messages array if not exists
            if not state.messages:
                state.messages = []
                
            # Add current user input and assistant response to conversation history
            # Note: Conversation memory is now managed by custom ConversationMemoryManager
            state.messages.append({"role": "user", "content": state.user_input})
            state.messages.append({"role": "assistant", "content": state.generated_answer})
            state.logs = (state.logs or []) + ["✅ Generated contextual answer with conversation history"]

        except Exception as e:
            logger.exception("Answer generation failed; issuing fallback response", exc_info=e)
            fallback_answer = {
                "answer": "I don't know based on available knowledge.",
                "citations": [],
                "is_idk": True,
                "confidence": "low",
                "error": str(e),
            }
            state.generated_answer = json.dumps(fallback_answer, ensure_ascii=False)
            state.is_idk_response = True
            state.logs = (state.logs or []) + [f"⚠️ Fallback answer used due to generation failure: {e}"]

        return state

    def human_review_node(self, state: KnowledgeState):
        """
        Handle human-in-the-loop review and validation of AI-generated answers.
        
        This node manages the human review process, supporting various
        feedback types including approval, rejection, and manual editing.
        Implements a LangGraph interrupt to pause execution for human input when
        no explicit feedback is present. Falls back to auto-approval in
        non-interactive scenarios if interrupt is unavailable.
        
        Args:
            state: KnowledgeState object containing generated_answer and optional feedback
            
        Returns:
            KnowledgeState: Updated state with final_answer and feedback status
            
        Feedback Types:
            - "approved": Answer meets quality standards
            - "rejected": Answer doesn't meet requirements
            - "edited": Answer was manually modified
            - None: Auto-approval (default behavior)
            
        Processing:
            - Checks for explicit human feedback
            - Implements auto-approval for batch processing
            - Handles rejection and editing scenarios
            - Updates final_answer based on feedback
            - Logs review decisions and actions
        """
        try:
            # Check for explicit human feedback
            feedback = getattr(state, "human_feedback", None)

            # Determine whether human review is required
            require_review = getattr(state, "require_human_review", None)
            if require_review is None:
                # Infer from knowledge_type when not explicitly set
                require_review = getattr(state, "knowledge_type", "conversational") in ("reusable", "verified")

            # If review is required, no feedback yet, and interrupt is available, pause the graph here
            if require_review and not feedback and interrupt is not None:
                # Message guides the API/UI to provide the required fields back
                _ = interrupt(
                    "Awaiting human review: set 'human_feedback' to one of"
                    " ['approved','rejected','edited'] and optionally set"
                    " 'edited_answer' when feedback is 'edited'."
                )
                # Execution will resume here once feedback is supplied via the graph resume
                feedback = getattr(state, "human_feedback", None)

            if not feedback:
                # Auto-approval path when running headless or interrupt not available
                feedback = "approved"
                state.final_answer = state.generated_answer
                state.logs = (state.logs or []) + ["✅ Auto-approved answer"]

            elif feedback == "rejected":
                # Handle rejection - no final answer
                state.final_answer = None
                state.logs = (state.logs or []) + ["❌ Answer was rejected by human"]

            elif feedback == "edited":
                # Handle manual editing
                if hasattr(state, "edited_answer") and state.edited_answer:
                    state.final_answer = state.edited_answer
                    state.logs = (state.logs or []) + ["✏️ Human edited the answer"]
                else:
                    state.final_answer = None
                    state.logs = (state.logs or []) + ["⚠️ Edit requested but no edited_answer provided"]

            else:
                # Unknown feedback type - default to generated answer
                state.final_answer = state.generated_answer
                state.logs = (state.logs or []) + [f"⚠️ Unknown feedback '{feedback}', defaulting to generated answer"]

            # Save feedback for traceability
            state.human_feedback = feedback

        except Exception as e:
            # Error handling - default to generated answer
            state.final_answer = state.generated_answer
            state.human_feedback = "error"
            state.logs = (state.logs or []) + [f"❌ Human review failed: {e}"]

        return state

    def validated_store_node(self, state: KnowledgeState):
        """
        Intelligent storage routing based on knowledge type and human feedback.
        
        This node implements smart storage decisions:
        - Conversational content: Stored in checkpoint memory only
        - Reusable/verified knowledge: Stored in both checkpoint and vector DB
        
        Args:
            state: KnowledgeState object containing final_answer, feedback, and knowledge_type
            
        Returns:
            KnowledgeState: Updated state with storage status
            
        Storage Logic:
            - Default: Store in checkpoint memory (conversational)
            - knowledge_type='reusable' or 'verified': Also store in vector DB
            - Rejected content: Not stored anywhere
        """
        logger.info(f"🔍 validated_store_node called with knowledge_type: {getattr(state, 'knowledge_type', 'None')}")
        logger.info(f"🔍 final_answer: {getattr(state, 'final_answer', 'None')}")
        logger.info(f"🔍 human_feedback: {getattr(state, 'human_feedback', 'None')}")
        try:
            # Check for answer to validate
            if not state.final_answer:
                logger.info("⚠️ No final_answer to validate, skipping")
                state.logs = (state.logs or []) + ["⚠️ No final_answer to validate, skipping"]
                return state

            # Check feedback status
            if state.human_feedback not in ("approved", "edited"):
                logger.info(f"ℹ️ Skipping validation because feedback = {state.human_feedback}")
                state.logs = (state.logs or []) + [f"ℹ️ Skipping validation because feedback = {state.human_feedback}"]
                return state

            # Determine knowledge type (default to conversational)
            knowledge_type = getattr(state, "knowledge_type", "conversational")
            
            # Always store in checkpoint memory for conversation continuity
            # Mark as validated at this point; downstream vector storage may refine status
            state.status = "validated"
            state.logs = (state.logs or []) + ["✅ Conversation history managed by LangGraph checkpoint memory (thread-isolated)"]
            
            # Check if content should also be stored in vector DB
            if knowledge_type in ("reusable", "verified"):
                try:
                    # Create a document for vector storage
                    from langchain.schema import Document
                    
                    # Prepare metadata for vector storage
                    metadata = {
                        "source": f"human_approved_{state.query_type}",
                        "thread_id": getattr(state, "thread_id", "unknown"),
                        "knowledge_type": knowledge_type,
                        "human_feedback": state.human_feedback,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_input": state.user_input or "unknown_query"
                    }
                    
                    # Add any existing metadata
                    if state.metadata:
                        metadata.update(state.metadata)
                    
                    # Create document for vector storage
                    doc = Document(
                        page_content=state.final_answer,
                        metadata=metadata
                    )
                    
                    # Store in vector database
                    if self.vectorstore:
                        logger.info(f"📚 Attempting to store {knowledge_type} knowledge in vector DB...")
                        logger.info(f"📚 Document content: {doc.page_content[:100]}...")
                        logger.info(f"📚 Document metadata: {doc.metadata}")
                        self.vectorstore.add_documents([doc])
                        state.logs = (state.logs or []) + [f"✅ {knowledge_type.title()} knowledge stored in vector database"]
                        # Refine status to indicate successful vector storage
                        state.status = "stored"
                        logger.info(f"📚 Successfully stored {knowledge_type} knowledge in vector DB")
                    else:
                        state.logs = (state.logs or []) + ["⚠️ Vector store not available, skipping vector storage"]
                        # Refine status to indicate validation without vector storage
                        state.status = "validated_no_store"
                        logger.warning("⚠️ Vector store not available, skipping vector storage")
                        
                except Exception as e:
                    state.logs = (state.logs or []) + [f"❌ Failed to store in vector DB: {e}"]
                    # Refine status to indicate vector storage failure after validation
                    state.status = "store_failed"
                    logger.error(f"Failed to store knowledge in vector DB: {e}")
            else:
                # Conversational content - checkpoint memory only
                state.logs = (state.logs or []) + ["💬 Conversational content stored in checkpoint memory only"]

        except Exception as e:
            state.status = "error"
            state.logs = (state.logs or []) + [f"❌ Validation failed: {e}"]

        return state

    # =============================================================================
    # GRAPH COMPILATION
    # =============================================================================

    def build(self) -> CompiledStateGraph:
        """
        Build and compile the complete LangGraph workflow.
        
        This method constructs the workflow graph by defining nodes,
        setting up conditional routing, and establishing the flow
        between different processing stages.
        
        Returns:
            CompiledStateGraph: Compiled and checkpointed workflow graph
            
        Graph Structure:
            Entry Point: router
            Conditional Routing: Based on query_type
            Ingestion Branch: router -> chunk -> embed -> store -> END
            Query Branch: router -> retriever -> answer -> review -> validated_store -> END
            
        Features:
            - Conditional edges for workflow routing
            - Checkpointing for state persistence
            - Error handling and logging throughout
            - Human-in-the-loop integration
        """
        # Create the base state graph
        graph = StateGraph(KnowledgeState)

        # =============================================================================
        # NODE DEFINITIONS
        # =============================================================================
        
        # Add all workflow nodes
        graph.add_node("router", self.input_router)           # Entry point and routing
        graph.add_node("preprocess", self.preprocess_node)    # Optional preprocessing
        graph.add_node("chunk", self.chunk_doc_node)          # Document chunking
        graph.add_node("embed", self.embed_node)              # Embedding generation
        graph.add_node("store", self.store_node)              # Document storage
        graph.add_node("retriever", self.retriever_node)      # Document retrieval
        graph.add_node("answer", self.answer_gen_node)        # Answer generation
        graph.add_node("review", self.human_review_node)      # Human review
        graph.add_node("validated_store", self.validated_store_node)  # Validated storage

        # Set the entry point
        graph.set_entry_point("router")

        # =============================================================================
        # CONDITIONAL ROUTING
        # =============================================================================
        
        # Define routing logic based on query_type
        def route_condition(state):
            if hasattr(state, 'query_type'):
                if state.query_type == "ingest":
                    return "ingest"
                elif state.query_type == "query":
                    return "query"
                else:
                    return "__end__"
            return "query"
        
        # Add conditional edges from router
        graph.add_conditional_edges("router", route_condition, {
            "ingest": "preprocess",      # Route to document ingestion
            "query": "retriever",   # Route to knowledge query
            "__end__": END,         # End workflow for invalid types
        })

        # =============================================================================
        # INGESTION BRANCH
        # =============================================================================
        
        # Document processing pipeline
        graph.add_edge("preprocess", "chunk")
        graph.add_edge("chunk", "embed")      # Chunk -> Embed
        graph.add_edge("embed", "store")     # Embed -> Store
        graph.add_edge("store", END)         # Store -> End

        # =============================================================================
        # QUERY BRANCH
        # =============================================================================
        
        # Knowledge query pipeline
        graph.add_edge("retriever", "answer")           # Retrieve -> Generate
        graph.add_edge("answer", "review")             # Generate -> Review
        graph.add_edge("review", "validated_store")    # Review -> Store
        graph.add_edge("validated_store", END)         # Store -> End

        # Compile the graph with checkpointing
        return graph.compile(checkpointer=self.checkpointer)