"""
Knowledge State Management for Smart Second Brain LangGraph Workflows.

This module defines the core state object that tracks the entire lifecycle of
knowledge processing in the Smart Second Brain system. The KnowledgeState class
serves as the central data structure that flows through all LangGraph nodes,
maintaining context, data, and metadata throughout document ingestion and query workflows.

Key Features:
- Comprehensive state tracking for document ingestion workflows
- Conversation context management for query workflows
- Metadata preservation and organization
- Human-in-the-loop feedback integration
- Debug logging and status tracking
- Flexible metadata storage for various document types

State Flow:
1. Document Ingestion: raw_document -> chunks -> embeddings -> storage
2. Knowledge Query: user_input -> retrieval -> generation -> validation
3. Feedback Loop: human_feedback -> edits -> final_answer

Usage:
    # For document ingestion
    state = KnowledgeState(
        query_type="ingest",
        raw_document="Your document content...",
        source="webpage",
        categories=["research", "ai"]
    )
    
    # For knowledge query
    state = KnowledgeState(
        query_type="query",
        user_input="What are the main concepts?",
        messages=[{"role": "user", "content": "What are the main concepts?"}]
    )

Author: Smart Second Brain Team
Version: 0.1.0
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class KnowledgeState(BaseModel):
    """
    State object for the Smart Second Brain LangGraph workflows.
    
    This class serves as the central data structure that flows through all
    LangGraph nodes, maintaining context, data, and metadata throughout
    document ingestion and query workflows. It provides a comprehensive
    representation of the system's knowledge processing state.
    
    The state object is designed to be:
    - Immutable during node execution (LangGraph requirement)
    - Rich in metadata for debugging and monitoring
    - Flexible for various document types and use cases
    - Extensible for future workflow enhancements
    
    Attributes:
        messages: Conversation-style messages for chat-like interactions
        user_input: Raw user input (query or document content)
        query_type: Workflow type identifier ('ingest' or 'query')
        raw_document: Full document text for ingestion workflows
        chunks: Text chunks after document splitting
        embeddings: Vector representations of text chunks
        source: Document source identifier
        categories: Document classification tags
        metadata: Additional structured metadata
        retrieved_docs: Retrieved documents from vector database
        retrieved_chunks: Raw text chunks from retrieval
        generated_answer: AI-generated response draft
        final_answer: Human-approved final response
        human_feedback: Human feedback on AI outputs
        edits: Manual corrections and edits
        conversation_history: Complete conversation context
        user_preferences: User customization settings
        status: Current workflow status
        logs: Debug and execution logs
    """

    # =============================================================================
    # CORE LANGRAPH MESSAGES
    # =============================================================================
    
    messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation-style messages exchanged between user, AI, and system."
    )
    """
    Conversation-style messages for chat-like interactions.
    
    This field maintains the conversation context in a format compatible with
    LangGraph's message passing system. Each message has a 'role' and 'content'
    field, similar to OpenAI's chat completion format.
    
    Example:
        [
            {"role": "user", "content": "Summarize my notes"},
            {"role": "assistant", "content": "Here's a summary..."},
            {"role": "user", "content": "Add more detail"}
        ]
    
    Used for:
    - Maintaining conversation context across workflow nodes
    - Providing conversation history to language models
    - Enabling multi-turn interactions and follow-up questions
    """

    # =============================================================================
    # USER INTERACTION
    # =============================================================================
    
    user_input: Optional[str] = Field(
        None, description="Raw input from user (query or new document)."
    )
    """
    Raw user input that initiates the workflow.
    
    For query workflows: The user's question or request
    For ingestion workflows: The document content to be processed
    
    This field serves as the primary input that triggers the workflow
    and is preserved throughout the entire processing pipeline.
    """
    
    query_type: Optional[str] = Field(
        None, description="Either 'ingest' for new document or 'query' for retrieval."
    )
    """
    Workflow type identifier that determines the processing path.
    
    Values:
        - 'ingest': Document ingestion workflow (chunking -> embedding -> storage)
        - 'query': Knowledge query workflow (retrieval -> generation -> validation)
    
    This field is used by workflow nodes to determine which processing
    logic to apply and which fields to populate.
    """

    # =============================================================================
    # DOCUMENT INGESTION
    # =============================================================================
    
    raw_document: Optional[str] = Field(
        None, description="Full raw text of the ingested document."
    )
    """
    Complete document content for ingestion workflows.
    
    This field contains the full, unprocessed text of the document
    that needs to be ingested into the knowledge base. It serves as
    the source material for the entire ingestion pipeline.
    
    Used by:
    - Text splitting nodes to create manageable chunks
    - Metadata extraction nodes for content analysis
    - Quality assessment nodes for document validation
    """
    
    chunks: Optional[List[str]] = Field(
        None, description="Text chunks after splitting for embedding."
    )
    """
    Text chunks created by splitting the raw document.
    
    These chunks are created by the text splitting node and are
    designed to be optimal for embedding generation. Each chunk
    should be semantically coherent and within token limits.
    
    Used by:
    - Embedding generation nodes for vector creation
    - Storage nodes for database insertion
    - Retrieval nodes for content matching
    """
    
    embeddings: Optional[List[List[float]]] = Field(
        None, description="Vector embeddings for the chunks."
    )
    """
    Vector representations of text chunks for semantic search.
    
    Each chunk is converted to a high-dimensional vector that
    captures its semantic meaning. These embeddings enable
    similarity-based retrieval and semantic search capabilities.
    
    Used by:
    - Vector database storage nodes
    - Similarity search and retrieval nodes
    - Clustering and organization nodes
    """

    # =============================================================================
    # METADATA FIELDS
    # =============================================================================
    
    source: Optional[str] = Field(
        None, description="Source of the document (e.g., filename, URL, transcript ID)."
    )
    """
    Document source identifier for provenance tracking.
    
    This field helps track where documents originated from,
    enabling source attribution, quality assessment, and
    content organization by source type.
    
    Examples:
        - "webpage: https://example.com/article"
        - "document: research_paper.pdf"
        - "transcript: meeting_2024_01_15"
        - "api: news_api_ingestion"
    """
    
    categories: Optional[List[str]] = Field(
        None, description="One or more categories/tags for the document."
    )
    """
    Document classification tags for organization and filtering.
    
    Categories help organize documents by topic, type, or domain,
    enabling better retrieval, filtering, and content organization.
    
    Examples:
        - ["ai", "machine_learning", "research"]
        - ["business", "strategy", "planning"]
        - ["technical", "documentation", "api"]
        - ["news", "technology", "trends"]
    """
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (author, date, page, etc.)."
    )
    """
    Flexible metadata storage for document attributes.
    
    This field provides a flexible way to store additional
    document information that doesn't fit into predefined fields.
    It's extensible and can accommodate various document types.
    
    Common metadata:
        - author: Document author or creator
        - date: Creation or modification date
        - page: Page numbers for multi-page documents
        - version: Document version or revision
        - language: Document language
        - confidence: Quality or confidence score
    """

    # =============================================================================
    # RETRIEVAL & ANSWERING
    # =============================================================================
    
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(
        None, description="Relevant documents retrieved from vector DB."
    )
    """
    Retrieved documents from the vector database.
    
    This field contains the documents retrieved during the
    query workflow, including their content, metadata, and
    relevance scores. It serves as the foundation for
    answer generation and validation.
    
    Structure:
        [
            {
                "content": "Document text content...",
                "metadata": {"source": "...", "categories": [...]},
                "score": 0.95,
                "chunk_id": "chunk_123"
            }
        ]
    """
    
    retrieved_chunks: Optional[List[str]] = Field(
        None, description="Raw chunks retrieved from the vector DB."
    )
    """
    Raw text chunks from retrieved documents.
    
    These are the actual text chunks that matched the query,
    extracted from the retrieved documents. They provide the
    raw content for answer generation and context building.
    
    Used by:
    - Answer generation nodes for context building
    - Validation nodes for content verification
    - Human review nodes for quality assessment
    """
    
    generated_answer: Optional[str] = Field(
        None, description="AI-generated draft answer or summary."
    )
    """
    AI-generated response draft for the user's query.
    
    This field contains the initial answer generated by the
    language model based on the retrieved documents. It serves
    as a starting point for human review and refinement.
    
    Used by:
    - Human review nodes for quality assessment
    - Validation nodes for answer verification
    - Feedback collection nodes for improvement
    """
    
    final_answer: Optional[str] = Field(
        None, description="Final, human-approved answer or summary."
    )
    """
    Final, approved answer after human review and validation.
    
    This field contains the final answer that has been
    reviewed, approved, or edited by a human. It represents
    the authoritative response to the user's query.
    
    Used by:
    - Response delivery nodes for user communication
    - Feedback collection nodes for quality metrics
    - Learning nodes for system improvement
    """

    # =============================================================================
    # HUMAN-IN-THE-LOOP
    # =============================================================================
    
    human_feedback: Optional[str] = Field(
        None, description="Feedback on AI output: 'approved' | 'rejected' | 'edited'."
    )
    """
    Human feedback on AI-generated outputs.
    
    This field tracks the human's assessment of AI-generated
    content, enabling quality control and system improvement.
    
    Values:
        - 'approved': Content meets quality standards
        - 'rejected': Content doesn't meet requirements
        - 'edited': Content was modified by human
        - 'needs_review': Content requires further review
    """
    
    edits: Optional[str] = Field(
        None, description="Manual corrections provided by the human."
    )
    """
    Manual corrections and improvements made by humans.
    
    This field stores the specific changes or improvements
    made by human reviewers, enabling learning and system
    improvement over time.
    
    Used for:
    - Training data for model improvement
    - Quality metrics and benchmarking
    - Process optimization and automation
    """
    
    edited_answer: Optional[str] = Field(
        None, description="The edited version of the answer provided by human feedback."
    )
    """
    The edited version of the answer after human review.
    
    This field stores the final edited answer that will be used
    as the final_answer when human feedback is "edited".
    
    Used for:
    - Storing human-edited content for final answer
    - Workflow processing of edited feedback
    - Quality control and validation
    """
    
    knowledge_type: Optional[str] = Field(
        None, description="Type of knowledge: 'conversational' | 'reusable' | 'verified'."
    )
    """
    Classification of the knowledge content for storage decisions.
    
    This field determines how approved content should be stored:
    - 'conversational': Chat history only (default)
    - 'reusable': Can be stored in vector DB for future retrieval
    - 'verified': High-quality knowledge that should be embedded and stored
    
    Used for:
    - Storage routing decisions
    - Knowledge base management
    - Content lifecycle management
    """

    # =============================================================================
    # MEMORY & CONTEXT
    # =============================================================================
    
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, description="Running log of user/AI/system messages for context."
    )
    """
    Complete conversation history for context maintenance.
    
    This field maintains a comprehensive log of all interactions
    in the current session, enabling context-aware responses
    and multi-turn conversations.
    
    Structure:
        [
            {"role": "user", "content": "Initial question", "timestamp": "..."},
            {"role": "assistant", "content": "Answer", "timestamp": "..."},
            {"role": "user", "content": "Follow-up", "timestamp": "..."}
        ]
    """
    
    user_preferences: Optional[Dict[str, Any]] = Field(
        None, description="User customization, e.g. summary style, tone, etc."
    )
    """
    User customization preferences for personalization.
    
    This field stores user-specific preferences that influence
    how content is processed, formatted, and delivered.
    
    Common preferences:
        - summary_style: "concise" | "detailed" | "bullet_points"
        - tone: "professional" | "casual" | "academic"
        - language: "en" | "es" | "fr" | "de"
        - detail_level: "high" | "medium" | "low"
    """

    # =============================================================================
    # CONTROL & DEBUG
    # =============================================================================
    
    status: Optional[str] = Field(
        None, description="Pipeline status: 'pending', 'processing', 'done', 'error'."
    )
    """
    Current workflow execution status.
    
    This field tracks the progress of the workflow through
    various stages, enabling monitoring, debugging, and
    user feedback.
    
    Values:
        - 'pending': Workflow is queued for execution
        - 'processing': Workflow is currently running
        - 'done': Workflow completed successfully
        - 'error': Workflow encountered an error
        - 'paused': Workflow is paused for human input
    """
    
    logs: Optional[List[str]] = Field(
        default_factory=list,
        description="Debug logs collected during pipeline execution."
    )
    """
    Debug and execution logs for troubleshooting.
    
    This field collects log messages from various workflow
    nodes, enabling debugging, monitoring, and performance
    analysis.
    
    Used for:
    - Debugging workflow issues
    - Performance monitoring and optimization
    - Audit trails and compliance
    - User support and troubleshooting
    """
    