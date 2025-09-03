"""
FastAPI routes for LangGraph integration

This module provides REST API endpoints to interact with the Smart Second Brain
LangGraph workflows for document ingestion and querying.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Depends

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agentic.workflows.master_graph_builder import MasterGraphBuilder
from agentic.core.knowledge_state import KnowledgeState
from shared.utils.logging_config import setup_logging
from api.core.config import settings

# Set up logging
logger = setup_logging("graph_api")

# Create router
router = APIRouter(prefix="/api/v1/graph", tags=["graph"])

# Global graph instance (in production, use dependency injection)
graph_builder = None
compiled_graph = None


def get_graph_builder() -> MasterGraphBuilder:
    """Dependency to get or create graph builder instance."""
    global graph_builder, compiled_graph

    if graph_builder is None:
        # Initialize with environment-based configuration
        api_key = settings.openai_api_key
        azure_endpoint = settings.azure_openai_endpoint_url

        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )

        try:
            # Initialize with proper models
            from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
            
            # Initialize Azure embedding model
            embedding_model = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-small",  # Your embedding deployment name
                openai_api_version="2024-12-01-preview",
                azure_endpoint=azure_endpoint,
                openai_api_key=api_key
            )
            
            # Initialize Azure LLM
            llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",  # Your LLM deployment name (from .env MODEL_NAME)
                openai_api_version="2024-12-01-preview",
                azure_endpoint=azure_endpoint,
                openai_api_key=api_key,
                temperature=0.1
            )
            
            # Initialize vectorstore
            from langchain_chroma import Chroma
            vectorstore = Chroma(
                collection_name="smart_second_brain",
                embedding_function=embedding_model,
                persist_directory="./chroma_db"
            )
            
            graph_builder = MasterGraphBuilder(
                llm=llm,
                embedding_model=embedding_model,
                vectorstore=vectorstore,
                chromadb_dir="./chroma_db"
            )
            compiled_graph = graph_builder.build()
            logger.info("✅ Graph builder initialized with all models")
        except Exception as e:
            logger.error(f"❌ Failed to initialize graph builder: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize graph: {str(e)}"
            )

    return graph_builder


# Pydantic models for request/response
class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    document: str = Field(..., description="The document content to ingest")
    source: Optional[str] = Field(None, description="Source of the document")
    categories: Optional[List[str]] = Field(None, description="Categories for the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "document": "Machine learning is a subset of artificial intelligence...",
                "source": "textbook",
                "categories": ["ai", "machine_learning"],
                "metadata": {"author": "John Doe", "chapter": "Introduction"}
            }
        }


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., description="The question to ask")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")

    class Config:
        schema_extra = {
            "example": {
                "query": "What are the key features of artificial intelligence?",
                "thread_id": "conversation_123"
            }
        }


class WorkflowResponse(BaseModel):
    """Response model for workflow results."""
    success: bool
    thread_id: str
    result: Dict[str, Any]
    execution_time: float
    timestamp: datetime

    class Config:
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
    """Response model for health checks."""
    status: str
    graph_initialized: bool
    vectorstore_ready: bool
    embedding_model_ready: bool
    llm_ready: bool
    timestamp: datetime


@router.post("/ingest", response_model=WorkflowResponse)
async def ingest_document(
    request: IngestRequest,
    graph_builder: MasterGraphBuilder = Depends(get_graph_builder)
):
    """
    Ingest a document into the knowledge base.

    This endpoint triggers the document ingestion workflow:
    chunking -> embedding -> storage
    """
    start_time = datetime.utcnow()

    try:
        # Generate thread ID if not provided
        thread_id = f"ingest_{int(start_time.timestamp())}"

        # Create knowledge state
        state = KnowledgeState(
            query_type="ingest",
            raw_document=request.document,
            source=request.source or "api_ingest",
            categories=request.categories or [],
            metadata=request.metadata or {}
        )

        # Get compiled graph
        global compiled_graph
        if compiled_graph is None:
            compiled_graph = graph_builder.build()

        # Execute ingestion workflow
        logger.info(f"🔄 Starting document ingestion workflow: {thread_id}")
        result = compiled_graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

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
    Query the knowledge base for answers.

    This endpoint triggers the query workflow:
    retrieval -> answer generation -> review -> validation
    """
    start_time = datetime.utcnow()

    try:
        # Generate thread ID if not provided
        thread_id = request.thread_id or f"query_{int(start_time.timestamp())}"

        # Create knowledge state
        state = KnowledgeState(
            query_type="query",
            user_input=request.query,
            categories=[],
            messages=[{"role": "user", "content": request.query}]
        )

        # Get compiled graph
        global compiled_graph
        if compiled_graph is None:
            compiled_graph = graph_builder.build()

        # Execute query workflow
        logger.info(f"🔍 Starting query workflow: {thread_id}")
        result = compiled_graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

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
    """Check the health status of the graph and its components."""
    try:
        global compiled_graph

        # Check graph initialization
        graph_ready = compiled_graph is not None
        vectorstore_ready = graph_builder.vectorstore is not None
        embedding_ready = graph_builder.embedding_model is not None
        llm_ready = graph_builder.llm is not None

        # Determine overall status
        if all([graph_ready, vectorstore_ready, embedding_ready, llm_ready]):
            status = "healthy"
        elif any([graph_ready, vectorstore_ready, embedding_ready, llm_ready]):
            status = "degraded"
        else:
            status = "unhealthy"

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
        return HealthResponse(
            status="error",
            graph_initialized=False,
            vectorstore_ready=False,
            embedding_model_ready=False,
            llm_ready=False,
            timestamp=datetime.utcnow()
        )
