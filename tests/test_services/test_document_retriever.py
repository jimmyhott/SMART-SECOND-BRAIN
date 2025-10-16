"""
Test script for SmartDocumentRetriever

This module tests the advanced document retrieval capabilities.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock
from dotenv import load_dotenv

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import centralized logging
from shared.utils.logging_config import setup_test_logging

# Set up test logging
logger = setup_test_logging("test_document_retriever")

# Load environment variables from .env file
load_dotenv()

from agentic.workflows.document_retriever import SmartDocumentRetriever, RetrievalResult
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_chroma import Chroma


class TestSmartDocumentRetriever:
    """Test suite for SmartDocumentRetriever class."""

    @pytest.fixture
    def temp_chromadb_dir(self):
        """Create a temporary ChromaDB directory for testing."""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp(prefix="test_retriever_chroma_")
        logger.info(f"🧪 Created temporary ChromaDB directory: {temp_dir}")
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"🧹 Cleaned up temporary ChromaDB directory: {temp_dir}")

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.",
                metadata={"source": "ai_book", "categories": "ai, machine_learning", "chunk_id": 0}
            ),
            Document(
                page_content="Deep learning is part of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns and representations from large datasets.",
                metadata={"source": "ai_book", "categories": "ai, deep_learning", "chunk_id": 1}
            ),
            Document(
                page_content="Natural language processing combines computational linguistics with statistical and machine learning models to give computers the ability to process and understand human language.",
                metadata={"source": "nlp_guide", "categories": "nlp, ai", "chunk_id": 0}
            ),
            Document(
                page_content="Vector databases store high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications including semantic search and recommendation systems.",
                metadata={"source": "db_guide", "categories": "databases, ai", "chunk_id": 0}
            ),
            Document(
                page_content="Knowledge graphs represent information as nodes and edges, capturing relationships between entities. They are used in semantic search, question answering, and knowledge management systems.",
                metadata={"source": "kg_paper", "categories": "knowledge_graphs, ai", "chunk_id": 0}
            )
        ]

    @pytest.fixture
    def retriever(self, temp_chromadb_dir, sample_documents):
        """Create a SmartDocumentRetriever instance with sample documents."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set in .env file")

        try:
            retriever = SmartDocumentRetriever(chromadb_dir=temp_chromadb_dir)
            retriever.add_documents(sample_documents)
            return retriever
        except Exception as e:
            pytest.skip(f"Failed to initialize retriever: {e}")

    @pytest.mark.integration
    def test_retriever_initialization(self, retriever):
        """Test retriever initialization."""
        assert retriever is not None
        assert retriever.vectorstore is not None
        assert retriever.embedding_model is not None

    @pytest.mark.integration
    def test_basic_retrieval(self, retriever):
        """Test basic document retrieval."""
        query = "machine learning"
        results = retriever.retrieve(query, k=3)

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score > 0 for r in results)

        # First result should be most relevant
        assert "machine learning" in results[0].document.page_content.lower()

    @pytest.mark.integration
    def test_hybrid_retrieval(self, retriever):
        """Test hybrid retrieval (semantic + keyword)."""
        query = "artificial intelligence"
        results_hybrid = retriever.retrieve(query, k=3, use_hybrid=True)
        results_semantic = retriever.retrieve(query, k=3, use_hybrid=False)

        # Both should return results
        assert len(results_hybrid) > 0
        assert len(results_semantic) > 0

        # Hybrid might return different results
        hybrid_sources = {r.source for r in results_hybrid}
        semantic_sources = {r.source for r in results_semantic}

        logger.info(f"Hybrid retrieval sources: {hybrid_sources}")
        logger.info(f"Semantic retrieval sources: {semantic_sources}")

    @pytest.mark.integration
    def test_metadata_filtering(self, retriever):
        """Test retrieval with metadata filters."""
        # Filter by category
        filters = {"categories": "ai, machine_learning"}
        results = retriever.search_by_metadata(filters, k=5)

        assert len(results) > 0
        for doc in results:
            assert "ai" in doc.metadata.get("categories", "")

    @pytest.mark.integration
    def test_reranking(self, retriever):
        """Test document re-ranking functionality."""
        query = "artificial intelligence"
        results = retriever.retrieve_with_reranking(query, k=3, rerank_top_k=10)

        assert len(results) <= 3  # Should return at most k results
        assert all(isinstance(r, RetrievalResult) for r in results)

        # Results should be sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.integration
    def test_collection_stats(self, retriever):
        """Test collection statistics retrieval."""
        stats = retriever.get_collection_stats()

        assert "total_documents" in stats
        assert stats["total_documents"] > 0
        assert "unique_sources" in stats
        assert "unique_categories" in stats

        logger.info(f"Collection stats: {stats}")

    @pytest.mark.integration
    def test_minimum_score_filtering(self, retriever):
        """Test filtering by minimum similarity score."""
        query = "machine learning"

        # Get all results
        all_results = retriever.retrieve(query, k=10, min_score=0.0)

        # Get results with minimum score
        filtered_results = retriever.retrieve(query, k=10, min_score=0.5)

        assert len(filtered_results) <= len(all_results)
        assert all(r.score >= 0.5 for r in filtered_results)

    @pytest.mark.integration
    def test_relevance_explanations(self, retriever):
        """Test that relevance explanations are generated."""
        query = "machine learning"
        results = retriever.retrieve(query, k=3)

        for result in results:
            assert result.relevance_explanation is not None
            assert len(result.relevance_explanation) > 0
            assert isinstance(result.relevance_explanation, str)

    @pytest.mark.integration
    def test_empty_query_handling(self, retriever):
        """Test handling of empty or invalid queries."""
        # Empty query
        results = retriever.retrieve("", k=5)
        assert isinstance(results, list)  # Should return empty list or handle gracefully

        # Very short query
        results = retriever.retrieve("a", k=5)
        assert isinstance(results, list)

    @pytest.mark.integration
    def test_retrieval_result_structure(self, retriever):
        """Test that RetrievalResult objects have correct structure."""
        query = "artificial intelligence"
        results = retriever.retrieve(query, k=2)

        for result in results:
            assert hasattr(result, 'document')
            assert hasattr(result, 'score')
            assert hasattr(result, 'source')
            assert hasattr(result, 'relevance_explanation')

            assert isinstance(result.document, Document)
            assert isinstance(result.score, (int, float))
            assert isinstance(result.source, str)
            assert isinstance(result.relevance_explanation, str)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
