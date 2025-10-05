"""
FastAPI routes for LangGraph integration in Smart Second Brain.

This module provides REST API endpoints to interact with the Smart Second Brain
LangGraph workflows for document ingestion and knowledge querying. It serves as
the bridge between the frontend interface and the AI-powered knowledge processing
system.

Key Endpoints:
- POST /ingest: Document ingestion workflow (chunking -> embedding -> storage)
- POST /query: Knowledge query workflow (retrieval -> generation -> validation)
- GET /health: System health status for all components

Core Functionality:
- LangGraph workflow integration
- Azure OpenAI model management
- ChromaDB vector store operations
- Thread-based conversation management
- Comprehensive error handling and logging

Dependencies:
- MasterGraphBuilder: Main workflow orchestrator
- KnowledgeState: LangGraph state management
- Azure OpenAI: Embeddings and language models
- ChromaDB: Vector database for document storage

Author: Smart Second Brain Team
Version: 0.1.0
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from io import BytesIO

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Request

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Add project root to Python path for proper imports
# This ensures we can import from agentic and shared modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# =============================================================================
# IMPORTS
# =============================================================================

# Core workflow components
from agentic.workflows.master_graph_builder import MasterGraphBuilder
from agentic.core.knowledge_state import KnowledgeState

# Utilities and configuration
from shared.utils.logging_config import setup_logging
from api.core.config import settings

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Set up dedicated logger for this module
logger = setup_logging("graph_api")

# =============================================================================
# ROUTER CONFIGURATION
# =============================================================================

# Create FastAPI router with versioning and tagging
router = APIRouter(prefix="/api/v1/graph", tags=["graph"])

# =============================================================================
# GLOBAL STATE MANAGEMENT
# =============================================================================

# Prefer app.state over module-level globals for better lifecycle management


def get_graph_builder(request: Request) -> MasterGraphBuilder:
    """
    Dependency function to get or create graph builder instance.
    
    This function implements lazy initialization of the MasterGraphBuilder,
    ensuring that expensive resources (LLM, embeddings, vectorstore) are only
    created when needed. It also handles configuration validation and error handling.
    
    Returns:
        MasterGraphBuilder: Configured and initialized graph builder instance
        
    Raises:
        HTTPException: If API key is missing or initialization fails
        
    Note:
        This is a global singleton pattern. In production, consider using
        dependency injection or a proper state management system.
    """
    # Use app.state; lazily initialize if missing
    graph_builder = getattr(request.app.state, "graph_builder", None)
    compiled_graph = getattr(request.app.state, "compiled_graph", None)

    if graph_builder is None:
        # Extract configuration from environment settings
        api_key = settings.openai_api_key
        azure_endpoint = settings.azure_openai_endpoint_url

        # Validate required configuration
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )

        try:
            # =============================================================================
            # MODEL INITIALIZATION
            # =============================================================================
            
            # Import Azure OpenAI models for embeddings and language generation
            from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
            
            # Initialize Azure embedding model for document vectorization
            embedding_model = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-small",  # Your embedding deployment name
                openai_api_version="2024-12-01-preview",    # Azure OpenAI API version
                azure_endpoint=azure_endpoint,              # Azure service endpoint
                openai_api_key=api_key                      # API key for authentication
            )
            
            # Initialize Azure language model for text generation and reasoning
            llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",                  # Your LLM deployment name
                openai_api_version="2024-12-01-preview",    # Azure OpenAI API version
                azure_endpoint=azure_endpoint,              # Azure service endpoint
                openai_api_key=api_key,                     # API key for authentication
                temperature=0.1                             # Low temperature for consistent outputs
            )
            
            # =============================================================================
            # VECTOR STORE INITIALIZATION
            # =============================================================================
            
            # Initialize ChromaDB vector store for document storage and retrieval
            from langchain_chroma import Chroma
            vectorstore = Chroma(
                collection_name="smart_second_brain",        # Collection name for documents
                embedding_function=embedding_model,          # Function to create embeddings
                persist_directory="./chroma_db"              # Local storage directory
            )
            
            # =============================================================================
            # GRAPH BUILDER INITIALIZATION
            # =============================================================================
            
            # Create and configure the main workflow orchestrator
            graph_builder = MasterGraphBuilder(
                llm=llm,                                    # Language model for reasoning
                embedding_model=embedding_model,             # Embedding model for vectorization
                vectorstore=vectorstore,                    # Vector database for storage
                chromadb_dir="./chroma_db"                  # ChromaDB storage directory
            )
            
            # Compile the workflow for execution
            compiled_graph = graph_builder.build()
            request.app.state.graph_builder = graph_builder
            request.app.state.compiled_graph = compiled_graph
            logger.info("✅ Graph builder initialized with all models")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize graph builder: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize graph: {str(e)}"
            )

    return graph_builder


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class IngestRequest(BaseModel):
    """
    Request model for document ingestion endpoint.
    
    This model defines the structure and validation rules for document ingestion
    requests, including the document content and optional metadata.
    
    Attributes:
        document: Required document content as text
        source: Optional source identifier (e.g., "webpage", "document")
        categories: Optional list of document categories for organization
        metadata: Optional dictionary of additional document metadata
    """
    document: str = Field(..., description="The document content to ingest")
    source: Optional[str] = Field(None, description="Source of the document")
    categories: Optional[List[str]] = Field(None, description="Categories for the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        """Pydantic configuration for request validation and documentation."""
        schema_extra = {
            "example": {
                "document": "Machine learning is a subset of artificial intelligence...",
                "source": "textbook",
                "categories": ["ai", "machine_learning"],
                "metadata": {"author": "John Doe", "chapter": "Introduction"}
            }
        }


class QueryRequest(BaseModel):
    """
    Request model for knowledge query endpoint.
    
    This model defines the structure for knowledge base queries, including
    the question and optional thread ID for conversation continuity.
    
    Attributes:
        query: Required question or query text
        thread_id: Optional thread ID for maintaining conversation context
        knowledge_type: Optional knowledge type guiding storage and review
        require_human_review: Optional flag to force human-in-the-loop interrupt
    """
    query: str = Field(..., description="The question to ask")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    knowledge_type: Optional[str] = Field(
        None,
        description="Knowledge type: 'conversational' | 'reusable' | 'verified'"
    )
    require_human_review: Optional[bool] = Field(
        None,
        description=(
            "If True, pause at review and wait for feedback; if False, do not pause; "
            "if None, infer from knowledge_type ('reusable'/'verified' -> pause)."
        )
    )

    class Config:
        """Pydantic configuration for request validation and documentation."""
        schema_extra = {
            "example": {
                "query": "What are the key features of artificial intelligence?",
                "thread_id": "conversation_123",
                "knowledge_type": "reusable",
                "require_human_review": True
            }
        }


class WorkflowResponse(BaseModel):
    """
    Response model for workflow execution results.
    
    This model provides a standardized response format for all workflow
    executions, including success status, thread ID, results, and timing.
    
    Attributes:
        success: Boolean indicating if the workflow completed successfully
        thread_id: Unique identifier for the workflow execution thread
        result: Dictionary containing workflow-specific results and data
        execution_time: Time taken to complete the workflow in seconds
        timestamp: UTC timestamp when the workflow completed
    """
    success: bool
    thread_id: str
    result: Dict[str, Any]
    execution_time: float
    timestamp: datetime

    class Config:
        """Pydantic configuration for response validation and documentation."""
        schema_extra = {
            "example": {
                "success": True,
                "thread_id": "thread_123",
                "result": {
                    "status": "validated",
                    "generated_answer": "AI systems have several key features...",
                    "retrieved_docs": [{"content": "AI content...", "metadata": {}}]
                },
                "execution_time": 2.34,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for system health status.
    
    This model provides comprehensive health information about all system
    components, enabling monitoring and troubleshooting.
    
    Attributes:
        status: Overall system health status (healthy/degraded/unhealthy)
        graph_initialized: Whether the LangGraph workflow is ready
        vectorstore_ready: Whether the vector database is accessible
        embedding_model_ready: Whether the embedding model is loaded
        llm_ready: Whether the language model is available
        timestamp: When the health check was performed
    """
    status: str
    graph_initialized: bool
    vectorstore_ready: bool
    embedding_model_ready: bool
    llm_ready: bool
    timestamp: datetime


class PDFIngestResult(BaseModel):
    """
    Individual PDF processing result.
    
    This model represents the result of processing a single PDF file,
    including success status, processing metrics, and any errors.
    
    Attributes:
        filename: Name of the processed PDF file
        success: Whether the PDF was processed successfully
        chunks_created: Number of text chunks created from the PDF
        thread_id: Unique identifier for the processing thread
        error: Error message if processing failed
        processing_time: Time taken to process this PDF in seconds
    """
    filename: str
    success: bool
    chunks_created: int = 0
    thread_id: Optional[str] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class MultiIngestResponse(BaseModel):
    """
    Response model for multiple PDF ingestion.
    
    This model provides comprehensive results for batch PDF processing,
    including individual file results and overall batch statistics.
    
    Attributes:
        success: Whether at least one PDF was processed successfully
        total_files: Total number of PDF files submitted
        processed_files: Number of PDFs processed successfully
        failed_files: Number of PDFs that failed processing
        results: List of individual PDF processing results
        execution_time: Total time for the entire batch operation
        timestamp: When the batch processing completed
    """
    success: bool
    total_files: int
    processed_files: int
    failed_files: int
    results: List[PDFIngestResult]
    execution_time: float
    timestamp: datetime

    class Config:
        """Pydantic configuration for response validation and documentation."""
        schema_extra = {
            "example": {
                "success": True,
                "total_files": 3,
                "processed_files": 2,
                "failed_files": 1,
                "results": [
                    {
                        "filename": "document1.pdf",
                        "success": True,
                        "chunks_created": 15,
                        "thread_id": "pdf_ingest_123_document1.pdf",
                        "processing_time": 2.34
                    },
                    {
                        "filename": "document2.pdf",
                        "success": False,
                        "error": "No text content found in PDF",
                        "processing_time": 0.0
                    }
                ],
                "execution_time": 5.67,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class FeedbackRequest(BaseModel):
    """
    Request model for human feedback submission.
    
    This model defines the structure for submitting human feedback on AI-generated
    answers, enabling human-in-the-loop validation and improvement.
    
    Attributes:
        thread_id: Required thread ID to identify the conversation
        feedback: Required feedback type (approved, rejected, edited)
        edits: Optional edited text when feedback is "edited"
        comment: Optional additional comments or notes
        knowledge_type: Optional knowledge classification for storage decisions
    """
    
    thread_id: str = Field(description="Thread ID to identify the conversation")
    feedback: str = Field(description="Feedback type: 'approved', 'rejected', or 'edited'")
    edits: Optional[str] = Field(None, description="Edited text when feedback is 'edited'")
    comment: Optional[str] = Field(None, description="Additional comments or notes")
    knowledge_type: Optional[str] = Field(None, description="Knowledge type: 'conversational', 'reusable', or 'verified'")

    class Config:
        """Pydantic configuration for request validation and documentation."""
        schema_extra = {
            "example": {
                "thread_id": "conversation_123",
                "feedback": "approved",
                "comment": "Great answer, very helpful!",
                "knowledge_type": "reusable"
            }
        }


class FeedbackResponse(BaseModel):
    """
    Response model for feedback submission.
    
    This model provides confirmation of feedback submission and any
    resulting actions taken by the system.
    
    Attributes:
        success: Whether the feedback was processed successfully
        thread_id: Thread ID the feedback was submitted for
        feedback: The feedback type that was submitted
        action_taken: Description of what action was taken based on feedback
        timestamp: When the feedback was processed
        message: Additional information or status message
    """
    
    success: bool = Field(description="Whether the feedback was processed successfully")
    thread_id: str = Field(description="Thread ID the feedback was submitted for")
    feedback: str = Field(description="The feedback type that was submitted")
    action_taken: str = Field(description="Description of what action was taken based on feedback")
    timestamp: datetime = Field(description="When the feedback was processed")
    message: str = Field(description="Additional information or status message")

    class Config:
        """Pydantic configuration for response validation and documentation."""
        schema_extra = {
            "example": {
                "success": True,
                "thread_id": "conversation_123",
                "feedback": "approved",
                "action_taken": "Answer approved and stored in knowledge base",
                "timestamp": "2024-01-15T10:30:00Z",
                "message": "Feedback processed successfully"
            }
        }


class FeedbackStatusResponse(BaseModel):
    """
    Response model for feedback status retrieval.
    
    This model provides information about the current feedback status
    for a given thread, including pending reviews and feedback history.
    
    Attributes:
        thread_id: Thread ID being queried
        has_pending_feedback: Whether there's a pending answer awaiting feedback
        current_answer: The current AI-generated answer awaiting review
        feedback_history: List of previous feedback submissions
        status: Current workflow status
    """
    
    thread_id: str = Field(description="Thread ID being queried")
    has_pending_feedback: bool = Field(description="Whether there's a pending answer awaiting feedback")
    current_answer: Optional[str] = Field(None, description="The current AI-generated answer awaiting review")
    feedback_history: List[Dict[str, Any]] = Field(default_factory=list, description="List of previous feedback submissions")
    status: str = Field(description="Current workflow status")

    class Config:
        """Pydantic configuration for response validation and documentation."""
        schema_extra = {
            "example": {
                "thread_id": "conversation_123",
                "has_pending_feedback": True,
                "current_answer": "Based on your documents, the main concepts are...",
                "feedback_history": [
                    {
                        "feedback": "approved",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "comment": "Great answer!"
                    }
                ],
                "status": "awaiting_feedback"
            }
        }


# =============================================================================
# PDF PROCESSING UTILITIES
# =============================================================================

async def extract_pdf_text(file: UploadFile) -> str:
    """
    Extract text content from PDF file.
    
    This function reads a PDF file and extracts all text content using
    PyPDF2 library. It handles multiple pages and provides error handling
    for corrupted or invalid PDF files.
    
    Args:
        file: UploadFile object containing the PDF data
        
    Returns:
        str: Extracted text content from the PDF
        
    Raises:
        HTTPException: If PDF text extraction fails
    """
    try:
        # Read file content
        content = await file.read()
        
        # Use PyPDF2 for text extraction
        import PyPDF2
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        
        # Extract text from all pages
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text.strip():
            raise ValueError("No text content found in PDF")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"PDF text extraction failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from PDF '{file.filename}': {str(e)}"
        )


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/ingest", response_model=WorkflowResponse)
async def ingest_document(
    request: IngestRequest,
    http_request: Request,
    graph_builder: MasterGraphBuilder = Depends(get_graph_builder)
):
    """
    Ingest a document into the knowledge base.
    
    This endpoint triggers the complete document ingestion workflow:
    1. Document chunking into manageable pieces
    2. Text embedding generation using Azure OpenAI
    3. Vector storage in ChromaDB for future retrieval
    4. Metadata indexing and categorization
    
    The workflow is executed asynchronously and returns a thread ID for tracking.
    
    Args:
        request: IngestRequest containing document content and metadata
        graph_builder: MasterGraphBuilder instance (injected dependency)
        
    Returns:
        WorkflowResponse: Execution results with thread ID and timing
        
    Raises:
        HTTPException: If ingestion fails or system is unavailable
        
    Example:
        POST /api/v1/graph/ingest
        {
            "document": "Your document content here...",
            "source": "webpage",
            "categories": ["research", "ai"],
            "metadata": {"author": "John Doe"}
        }
    """
    start_time = datetime.utcnow()

    try:
        # Generate unique thread ID for this ingestion workflow
        thread_id = f"ingest_{int(start_time.timestamp())}"

        # Create knowledge state for the ingestion workflow
        state = KnowledgeState(
            query_type="ingest",                           # Workflow type identifier
            raw_document=request.document,                 # Original document content
            source=request.source or "api_ingest",         # Document source or default
            categories=request.categories or [],           # Document categories
            metadata=request.metadata or {}                # Additional metadata
        )

        # Ensure compiled graph is available
        compiled_graph = getattr(http_request.app.state, "compiled_graph", None)
        if compiled_graph is None:
            compiled_graph = graph_builder.build()
            http_request.app.state.compiled_graph = compiled_graph

        # Execute the ingestion workflow with thread ID for tracking
        logger.info(f"🔄 Starting document ingestion workflow: {thread_id}")
        result = compiled_graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # Calculate execution time for performance monitoring
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"✅ Ingestion completed in {execution_time:.2f}s")
        return WorkflowResponse(
            success=True,
            thread_id=thread_id,
            result=result,
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(e)}"
        )


@router.post("/query", response_model=WorkflowResponse)
async def query_knowledge_base(
    request: QueryRequest,
    http_request: Request,
    graph_builder: MasterGraphBuilder = Depends(get_graph_builder)
):
    """
    Query the knowledge base for answers using natural language.
    
    This endpoint triggers the knowledge query workflow:
    1. Semantic search in the vector database
    2. Document retrieval and relevance scoring
    3. Answer generation using the language model
    4. Answer validation and quality assessment
    
    The workflow maintains conversation context through thread IDs.
    
    Args:
        request: QueryRequest containing the question and optional thread ID
        graph_builder: MasterGraphBuilder instance (injected dependency)
        
    Returns:
        WorkflowResponse: Query results with generated answer and retrieved documents
        
    Raises:
        HTTPException: If query execution fails or system is unavailable
        
    Example:
        POST /api/v1/graph/query
        {
            "query": "What are the main concepts discussed?",
            "thread_id": "conversation_123"
        }
    """
    start_time = datetime.utcnow()

    try:
        # Use provided thread ID or generate new one for conversation continuity
        thread_id = request.thread_id or f"query_{int(start_time.timestamp())}"

        # Create knowledge state for the query workflow
        # LangGraph checkpointer will handle conversation history automatically
        state = KnowledgeState(
            query_type="query",                            # Workflow type identifier
            user_input=request.query,                      # User's question
            categories=[],                                 # No categories for queries
            messages=[],                                   # Empty messages - LangGraph handles history
            knowledge_type=request.knowledge_type,
            require_human_review=request.require_human_review
        )

        # Ensure compiled graph is available
        compiled_graph = getattr(http_request.app.state, "compiled_graph", None)
        if compiled_graph is None:
            compiled_graph = graph_builder.build()
            http_request.app.state.compiled_graph = compiled_graph

        # Execute the query workflow with thread ID for context
        logger.info(f"🔍 Starting query workflow: {thread_id}")
        result = compiled_graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # LangGraph checkpointer automatically saves conversation state
        # No need for manual conversation memory management

        # Calculate execution time for performance monitoring
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"✅ Query completed in {execution_time:.2f}s")
        return WorkflowResponse(
            success=True,
            thread_id=thread_id,
            result=result,
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Check the health status of the graph system and its components.
    
    This endpoint provides comprehensive health information about all system
    components, enabling monitoring, alerting, and troubleshooting. It checks
    the availability and readiness of:
    - LangGraph workflow compilation
    - Vector database connectivity
    - Embedding model availability
    - Language model availability
    
    Returns:
        HealthResponse: Detailed health status for all components
        
    Note:
        This endpoint is designed to be called by monitoring systems and
        load balancers to determine service availability.
    """
    try:
        # =============================================================================
        # COMPONENT HEALTH CHECKS
        # =============================================================================
        
        # Check if the compiled workflow is available
        compiled_graph = getattr(request.app.state, "compiled_graph", None)
        graph_ready = compiled_graph is not None
        
        # Check if the vector store is accessible
        graph_builder = getattr(request.app.state, "graph_builder", None)
        vectorstore_ready = graph_builder is not None and graph_builder.vectorstore is not None
        
        # Check if the embedding model is loaded
        embedding_ready = graph_builder is not None and graph_builder.embedding_model is not None
        
        # Check if the language model is available
        llm_ready = graph_builder is not None and graph_builder.llm is not None

        # =============================================================================
        # OVERALL STATUS DETERMINATION
        # =============================================================================
        
        # Determine overall system health based on component status
        if all([graph_ready, vectorstore_ready, embedding_ready, llm_ready]):
            status = "healthy"          # All components are operational
        elif any([graph_ready, vectorstore_ready, embedding_ready, llm_ready]):
            status = "degraded"         # Some components are available
        else:
            status = "unhealthy"        # No components are available

        return HealthResponse(
            status=status,
            graph_initialized=graph_ready,
            vectorstore_ready=vectorstore_ready,
            embedding_model_ready=embedding_ready,
            llm_ready=llm_ready,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        # Return error status if health check itself fails
        return HealthResponse(
            status="error",
            graph_initialized=False,
            vectorstore_ready=False,
            embedding_model_ready=False,
            llm_ready=False,
            timestamp=datetime.utcnow()
        )


@router.post("/ingest-pdfs", response_model=MultiIngestResponse)
async def ingest_multiple_pdfs(
    http_request: Request,
    files: List[UploadFile] = File(...),
    source: str = Form(...),
    categories: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    graph_builder: MasterGraphBuilder = Depends(get_graph_builder)
):
    """
    Ingest multiple PDF files into the knowledge base.
    
    This endpoint processes multiple PDF files in sequence, extracting text,
    chunking content, and storing embeddings in the vector database. Each
    PDF is processed through the complete ingestion workflow with individual
    error handling and result tracking.
    
    Args:
        files: List of PDF files to process (required)
        source: Source identifier for all files (required)
        categories: Comma-separated categories (optional)
        author: Document author (optional)
        metadata: JSON string of additional metadata (optional)
        graph_builder: MasterGraphBuilder instance (injected dependency)
        
    Returns:
        MultiIngestResponse: Processing results for all files with detailed statistics
        
    Raises:
        HTTPException: If no files provided or system is unavailable
        
    Example:
        POST /api/v1/graph/ingest-pdfs
        Form data:
        - files: [file1.pdf, file2.pdf, file3.pdf]
        - source: "research_papers"
        - categories: "ai,research,machine_learning"
        - author: "Research Team"
        - metadata: {"department": "AI Lab", "year": "2024"}
    """
    start_time = datetime.utcnow()
    
    # Validate input
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Please upload at least one PDF file."
        )
    
    # Parse categories and metadata
    category_list = []
    if categories:
        category_list = [cat.strip() for cat in categories.split(",") if cat.strip()]
    
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON metadata provided: {metadata}")
            metadata_dict = {}
    
    # Add common metadata
    metadata_dict.update({
        "author": author or "Unknown",
        "source": source,
        "categories": category_list,
        "ingestion_date": datetime.utcnow().isoformat(),
        "batch_ingest": True
    })
    
    results = []
    processed_count = 0
    failed_count = 0
    
    # Ensure compiled graph is available
    compiled_graph = getattr(http_request.app.state, "compiled_graph", None)
    if compiled_graph is None:
        compiled_graph = graph_builder.build()
        http_request.app.state.compiled_graph = compiled_graph
    
    logger.info(f"🔄 Starting batch PDF ingestion: {len(files)} files")
    
    # Process each PDF file
    for file in files:
        file_start_time = datetime.utcnow()
        
        try:
            # Validate file type
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                results.append(PDFIngestResult(
                    filename=file.filename or "unknown",
                    success=False,
                    error="Invalid file type. Only PDF files are supported.",
                    processing_time=0.0
                ))
                failed_count += 1
                continue
            
            # Extract text from PDF
            pdf_content = await extract_pdf_text(file)
            
            # Create knowledge state for ingestion
            state = KnowledgeState(
                query_type="ingest",
                raw_document=pdf_content,
                source=f"{source}_{file.filename}",
                categories=category_list,
                metadata={**metadata_dict, "original_filename": file.filename}
            )
            
            # Generate unique thread ID for this PDF
            thread_id = f"pdf_ingest_{int(file_start_time.timestamp())}_{file.filename}"
            
            # Execute the ingestion workflow
            logger.info(f"📄 Processing PDF: {file.filename}")
            result = compiled_graph.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}}
            )
            
            # Calculate processing time for this file
            file_processing_time = (datetime.utcnow() - file_start_time).total_seconds()
            
            # Count chunks created
            chunks_created = len(result.get("chunks", []))
            
            results.append(PDFIngestResult(
                filename=file.filename,
                success=True,
                chunks_created=chunks_created,
                thread_id=thread_id,
                processing_time=file_processing_time
            ))
            processed_count += 1
            
            logger.info(f"✅ PDF processed successfully: {file.filename} ({chunks_created} chunks)")
            
        except Exception as e:
            file_processing_time = (datetime.utcnow() - file_start_time).total_seconds()
            error_msg = str(e)
            
            logger.error(f"❌ Failed to process PDF {file.filename}: {error_msg}")
            
            results.append(PDFIngestResult(
                filename=file.filename or "unknown",
                success=False,
                error=error_msg,
                processing_time=file_processing_time
            ))
            failed_count += 1
    
    # Calculate total execution time
    execution_time = (datetime.utcnow() - start_time).total_seconds()
    
    logger.info(f"🏁 Batch PDF ingestion completed: {processed_count} successful, {failed_count} failed in {execution_time:.2f}s")
    
    return MultiIngestResponse(
        success=processed_count > 0,
        total_files=len(files),
        processed_files=processed_count,
        failed_files=failed_count,
        results=results,
        execution_time=execution_time,
        timestamp=datetime.utcnow()
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    http_request: Request,
    graph_builder: MasterGraphBuilder = Depends(get_graph_builder)
):
    """
    Submit human feedback on AI-generated answers.
    
    This endpoint enables human-in-the-loop validation by allowing users to
    provide feedback on AI-generated answers. The feedback can be approval,
    rejection, or edits to improve the answer quality.
    
    Args:
        request: FeedbackRequest containing thread_id, feedback type, and optional edits
        graph_builder: MasterGraphBuilder dependency for workflow execution
        
    Returns:
        FeedbackResponse: Confirmation of feedback submission and actions taken
        
    Raises:
        HTTPException: If thread_id is invalid or feedback processing fails
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate feedback type
        valid_feedback_types = ["approved", "rejected", "edited"]
        if request.feedback not in valid_feedback_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback type. Must be one of: {valid_feedback_types}"
            )
        
        # Validate edits are provided when feedback is "edited"
        if request.feedback == "edited" and not request.edits:
            raise HTTPException(
                status_code=400,
                detail="Edits must be provided when feedback type is 'edited'"
            )
        
        # Validate knowledge type if provided
        if request.knowledge_type:
            valid_knowledge_types = ["conversational", "reusable", "verified"]
            if request.knowledge_type not in valid_knowledge_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid knowledge type. Must be one of: {valid_knowledge_types}"
                )
        
        logger.info(f"📝 Processing feedback for thread {request.thread_id}: {request.feedback}")
        
        # Get the compiled graph
        compiled_graph = getattr(http_request.app.state, "compiled_graph", None)
        if compiled_graph is None:
            compiled_graph = graph_builder.build()
            http_request.app.state.compiled_graph = compiled_graph
        
        # Create a new state with the feedback
        feedback_state = KnowledgeState(
            query_type="query",  # We're processing feedback on a query
            human_feedback=request.feedback,
            edits=request.edits,
            metadata={
                "feedback_comment": request.comment,
                "feedback_timestamp": start_time.isoformat(),
                "feedback_source": "api"
            }
        )
        
        # Process feedback through the workflow using LangGraph checkpointing
        try:
            # Get the current state from LangGraph checkpointing
            current_state = compiled_graph.get_state({"configurable": {"thread_id": request.thread_id}})
            
            if not current_state or not current_state.values:
                raise HTTPException(
                    status_code=404,
                    detail=f"No conversation found for thread_id: {request.thread_id}"
                )
            
            # Create a new state with the existing conversation data plus feedback
            existing_state = current_state.values
            logger.info(f"🔍 Creating feedback state with knowledge_type: {request.knowledge_type}")
            feedback_state = KnowledgeState(
                query_type=existing_state.get("query_type", "query"),
                user_input=existing_state.get("user_input"),
                messages=existing_state.get("messages", []),
                generated_answer=existing_state.get("generated_answer"),
                retrieved_docs=existing_state.get("retrieved_docs"),
                retrieved_chunks=existing_state.get("retrieved_chunks"),
                metadata=existing_state.get("metadata", {}),
                human_feedback=request.feedback,
                edits=request.edits,
                knowledge_type=request.knowledge_type,
                status=existing_state.get("status"),
                logs=existing_state.get("logs", [])
            )
            logger.info(f"🔍 Feedback state created with knowledge_type: {feedback_state.knowledge_type}")
            
            # Add feedback metadata
            if request.comment:
                feedback_state.metadata = feedback_state.metadata or {}
                feedback_state.metadata["feedback_comment"] = request.comment
                feedback_state.metadata["feedback_timestamp"] = start_time.isoformat()
                feedback_state.metadata["feedback_source"] = "api"
            
            # If feedback is "edited", update the edited_answer with the edits
            if request.feedback == "edited" and request.edits:
                feedback_state.edited_answer = request.edits
                logger.info(f"🔍 Updated edited_answer with edits: {request.edits[:100]}...")
            
            # Resume the graph from the interrupt by invoking with the updated state
            logger.info(f"🔄 Resuming workflow after human feedback")
            result = compiled_graph.invoke(
                feedback_state,
                config={"configurable": {"thread_id": request.thread_id}}
            )
            
            # Determine action taken based on feedback and knowledge type
            # Derive action from result
            if result.get("human_feedback") == "approved":
                if result.get("knowledge_type") in ("reusable", "verified"):
                    action_taken = f"Answer approved and stored as {result.get('knowledge_type')} knowledge in vector database"
                else:
                    action_taken = "Answer approved and stored in conversation history"
            elif result.get("human_feedback") == "rejected":
                action_taken = "Answer rejected, not stored in knowledge base"
            elif result.get("human_feedback") == "edited":
                if result.get("knowledge_type") in ("reusable", "verified"):
                    action_taken = f"Answer edited and stored as {result.get('knowledge_type')} knowledge in vector database"
                else:
                    action_taken = "Answer edited and stored in conversation history"
            else:
                action_taken = "Feedback processed"
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"✅ Feedback processed successfully in {execution_time:.2f}s")
            
            return FeedbackResponse(
                success=True,
                thread_id=request.thread_id,
                feedback=result.get("human_feedback", request.feedback),
                action_taken=action_taken,
                timestamp=datetime.utcnow(),
                message="Feedback processed successfully"
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing feedback: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process feedback: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in feedback submission: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/feedback/{thread_id}", response_model=FeedbackStatusResponse)
async def get_feedback_status(
    thread_id: str,
    http_request: Request,
    graph_builder: MasterGraphBuilder = Depends(get_graph_builder)
):
    """
    Get the current feedback status for a conversation thread.
    
    This endpoint provides information about whether there's a pending answer
    awaiting feedback and the history of feedback submissions for the thread.
    
    Args:
        thread_id: Thread ID to check feedback status for
        graph_builder: MasterGraphBuilder dependency (for consistency)
        
    Returns:
        FeedbackStatusResponse: Current feedback status and history
        
    Raises:
        HTTPException: If thread_id is invalid or status retrieval fails
    """
    try:
        logger.info(f"🔍 Checking feedback status for thread: {thread_id}")
        
        # Get the compiled graph
        compiled_graph = getattr(http_request.app.state, "compiled_graph", None)
        if compiled_graph is None:
            compiled_graph = graph_builder.build()
            http_request.app.state.compiled_graph = compiled_graph
        
        # Try to get the current state from LangGraph checkpointing
        try:
            # Get the current state from the compiled graph's checkpointing system
            current_state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
            
            if not current_state or not current_state.values:
                raise HTTPException(
                    status_code=404,
                    detail=f"No conversation found for thread_id: {thread_id}"
                )
            
            # Extract the KnowledgeState from the checkpoint
            state_values = current_state.values
            
            # Check if there's a pending answer awaiting feedback
            has_pending_feedback = (
                state_values.get("generated_answer") is not None and 
                state_values.get("human_feedback") is None
            )
            
            # Get current answer if pending
            current_answer = state_values.get("generated_answer") if has_pending_feedback else None
            
            # Build feedback history from conversation metadata
            feedback_history = []
            metadata = state_values.get("metadata", {})
            if metadata and "feedback_history" in metadata:
                feedback_history = metadata["feedback_history"]
            
            # Add current feedback if it exists
            if state_values.get("human_feedback"):
                current_feedback = {
                    "feedback": state_values.get("human_feedback"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "comment": metadata.get("feedback_comment") if metadata else None
                }
                feedback_history.append(current_feedback)
            
            # Determine current status
            if has_pending_feedback:
                status = "awaiting_feedback"
            elif state_values.get("human_feedback") == "approved":
                status = "approved"
            elif state_values.get("human_feedback") == "rejected":
                status = "rejected"
            elif state_values.get("human_feedback") == "edited":
                status = "edited"
            else:
                status = "unknown"
                
        except Exception as e:
            logger.warning(f"Could not retrieve state from LangGraph checkpointing: {e}")
            # If checkpointing fails, assume no pending feedback
            raise HTTPException(
                status_code=404,
                detail=f"No conversation found for thread_id: {thread_id}"
            )
            
            status = "conversation_exists"
        
        logger.info(f"✅ Feedback status retrieved: {status}")
        
        return FeedbackStatusResponse(
            thread_id=thread_id,
            has_pending_feedback=has_pending_feedback,
            current_answer=current_answer,
            feedback_history=feedback_history,
            status=status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving feedback status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve feedback status: {str(e)}"
        )
