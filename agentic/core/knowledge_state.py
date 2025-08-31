from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class KnowledgeState(BaseModel):
    """
    State object for the Smart Second Brain project.
    Tracks ingestion, retrieval, human-in-the-loop feedback, and memory.
    """

    # --- Core LangGraph messages (chat-like exchanges) ---
    messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation-style messages exchanged between user, AI, and system."
    )
    # Example: [{"role": "user", "content": "Summarize my notes"}]

    # --- User Interaction ---
    user_input: Optional[str] = Field(
        None, description="Raw input from user (query or new document)."
    )
    query_type: Optional[str] = Field(
        None, description="Either 'ingest' for new document or 'query' for retrieval."
    )

    # --- Document Ingestion ---
    raw_document: Optional[str] = Field(
        None, description="Full raw text of the ingested document."
    )
    chunks: Optional[List[str]] = Field(
        None, description="Text chunks after splitting for embedding."
    )
    embeddings: Optional[List[List[float]]] = Field(
        None, description="Vector embeddings for the chunks."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Metadata about the document (source, date, tags, etc.)."
    )

    # --- Retrieval & Answering ---
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(
        None, description="Relevant documents retrieved from vector DB."
    )
    generated_answer: Optional[str] = Field(
        None, description="AI-generated draft answer or summary."
    )
    final_answer: Optional[str] = Field(
        None, description="Final, human-approved answer or summary."
    )

    # --- Human-in-the-loop ---
    human_feedback: Optional[str] = Field(
        None, description="Feedback on AI output: 'approved' | 'rejected' | 'edited'."
    )
    edits: Optional[str] = Field(
        None, description="Manual corrections provided by the human."
    )

    # --- Memory & Context ---
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Running log of user/AI/system messages for context."
    )
    user_preferences: Optional[Dict[str, Any]] = Field(
        None, description="User customization, e.g. summary style, tone, etc."
    )

    # --- Control & Debug ---
    status: Optional[str] = Field(
        None, description="Pipeline status: 'pending', 'processing', 'done', 'error'."
    )
    logs: Optional[List[str]] = Field(
        None, description="Debug logs collected during pipeline execution."
    )