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
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Depends

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

# Global graph instances for application lifecycle management
# In production, consider using dependency injection or a proper state manager
graph_builder = None      # MasterGraphBuilder instance
compiled_graph = None     # Compiled LangGraph workflow


def get_graph_builder() -> MasterGraphBuilder:
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
    global graph_builder, compiled_graph

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
    """
    query: str = Field(..., description="The question to ask")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")

    class Config:
        """Pydantic configuration for request validation and documentation."""
        schema_extra = {
            "example": {
                "query": "What are the key features of artificial intelligence?",
                "thread_id": "conversation_123"
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


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.post("/ingest", response_model=WorkflowResponse)
async def ingest_document(
    request: IngestRequest,
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
        global compiled_graph
        if compiled_graph is None:
            compiled_graph = graph_builder.build()

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
        state = KnowledgeState(
            query_type="query",                            # Workflow type identifier
            user_input=request.query,                      # User's question
            categories=[],                                 # No categories for queries
            messages=[{"role": "user", "content": request.query}]  # Conversation history
        )

        # Ensure compiled graph is available
        global compiled_graph
        if compiled_graph is None:
            compiled_graph = graph_builder.build()

        # Execute the query workflow with thread ID for context
        logger.info(f"🔍 Starting query workflow: {thread_id}")
        result = compiled_graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

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
async def health_check(graph_builder: MasterGraphBuilder = Depends(get_graph_builder)):
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
        global compiled_graph

        # =============================================================================
        # COMPONENT HEALTH CHECKS
        # =============================================================================
        
        # Check if the compiled workflow is available
        graph_ready = compiled_graph is not None
        
        # Check if the vector store is accessible
        vectorstore_ready = graph_builder.vectorstore is not None
        
        # Check if the embedding model is loaded
        embedding_ready = graph_builder.embedding_model is not None
        
        # Check if the language model is available
        llm_ready = graph_builder.llm is not None

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
