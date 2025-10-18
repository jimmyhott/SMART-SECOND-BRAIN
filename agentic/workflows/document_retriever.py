"""
Document Retriever for Smart Second Brain

This module provides advanced document retrieval capabilities including:
- Semantic similarity search
- Hybrid retrieval (semantic + keyword)
- Metadata filtering
- Re-ranking and scoring
- Caching and performance optimization
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import centralized logging
from shared.utils.logging_config import setup_agentic_logging
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document, BaseRetriever
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
import numpy as np

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_agentic_logging()


@dataclass
class RetrievalResult:
    """Result from document retrieval with scoring."""
    document: Document
    score: float
    source: str
    relevance_explanation: str = ""


class SmartDocumentRetriever:
    """
    Advanced document retriever for Smart Second Brain.

    Features:
    - Vector similarity search with ChromaDB
    - Hybrid retrieval (semantic + keyword)
    - Metadata filtering
    - Query expansion
    - Re-ranking
    - Caching
    """

    def __init__(
        self,
        vectorstore: Chroma = None,
        embedding_model = None,
        collection_name: str = "smart_second_brain",
        chromadb_dir: str = "./chroma_db"
    ):
        """
        Initialize the document retriever.

        Args:
            vectorstore: Existing ChromaDB vectorstore
            embedding_model: Text embedding model
            collection_name: ChromaDB collection name
            chromadb_dir: Directory for ChromaDB persistence
        """
        self.collection_name = collection_name
        self.chromadb_dir = chromadb_dir

        # Initialize embedding model if not provided
        if embedding_model is None:
            embedding_model = self._initialize_embedding_model()

        self.embedding_model = embedding_model

        # Initialize or load vectorstore
        if vectorstore is None:
            self.vectorstore = self._initialize_vectorstore()
        else:
            self.vectorstore = vectorstore

        # Initialize retrievers
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.bm25_retriever = None  # Will be initialized with documents
        self.ensemble_retriever = None  # Will be initialized when needed

        logger.info("🔍 SmartDocumentRetriever initialized")

    def _initialize_embedding_model(self):
        """Initialize the embedding model based on environment configuration."""
        api_key = os.getenv("OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
        api_version = os.getenv("API_VERSION", "2024-12-01-preview")

        if azure_endpoint and azure_endpoint != "https://your-resource-name.openai.azure.com/":
            # Use Azure OpenAI
            deployment = "text-embedding-3-small"
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=deployment,
                openai_api_version=api_version,
                azure_endpoint=azure_endpoint,
                openai_api_key=api_key
            )
        else:
            # Use OpenAI directly
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=api_key
            )

        return embeddings

    def _initialize_vectorstore(self) -> Chroma:
        """Initialize or load ChromaDB vectorstore."""
        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.chromadb_dir
            )
            logger.info(f"💾 Loaded ChromaDB from {self.chromadb_dir}")
            return vectorstore
        except Exception as e:
            logger.warning(f"❌ Could not load ChromaDB: {e}")
            # Create new empty vectorstore
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.chromadb_dir
            )
            logger.info(f"🆕 Created new ChromaDB at {self.chromadb_dir}")
            return vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vectorstore.

        Args:
            documents: List of Document objects to add
        """
        try:
            # Add to vectorstore
            self.vectorstore.add_documents(documents)

            # Update BM25 retriever
            if self.bm25_retriever is None:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
            else:
                # Add new documents to existing BM25
                self.bm25_retriever.add_documents(documents)

            # Update ensemble retriever
            self._update_ensemble_retriever()

            logger.info(f"✅ Added {len(documents)} documents to retriever")

        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")

    def hydrate_bm25(self, documents: List[Document]) -> None:
        """Hydrate the BM25 retriever from existing documents without mutating the vectorstore."""
        if not documents:
            return

        try:
            if self.bm25_retriever is None:
                self.bm25_retriever = BM25Retriever.from_documents(documents)
            else:
                self.bm25_retriever.add_documents(documents)

            self._update_ensemble_retriever()
            logger.info("🔄 Hydrated BM25 retriever from persisted documents")
        except Exception as exc:
            logger.warning("BM25 hydration failed: %s", exc)

    def _update_ensemble_retriever(self):
        """Update the ensemble retriever with current retrievers."""
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]  # 70% semantic, 30% keyword
            )
            logger.info("🔄 Updated ensemble retriever")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve
            filters: Metadata filters to apply
            use_hybrid: Whether to use hybrid retrieval (semantic + keyword)
            min_score: Minimum similarity score threshold

        Returns:
            List of RetrievalResult objects with documents and scores
        """
        try:
            logger.info(f"🔍 Retrieving documents for: '{query}'")

            # Prepare search kwargs for the base vector retriever
            if filters:
                self.vector_retriever.search_kwargs.update({"filter": filters})
            self.vector_retriever.search_kwargs["k"] = k

            # Choose retrieval method
            if use_hybrid and self.ensemble_retriever:
                logger.info("🔄 Using hybrid retrieval (semantic + keyword)")
                results = self.ensemble_retriever.get_relevant_documents(query)
            else:
                logger.info("🔄 Using semantic retrieval only")
                results = self.vector_retriever.get_relevant_documents(query)

            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, doc in enumerate(results):
                # Calculate relevance score (simplified - in practice use actual similarity scores)
                score = max(1.0 - (i * 0.1), 0.1)  # Decreasing score

                result = RetrievalResult(
                    document=doc,
                    score=score,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_explanation=self._generate_relevance_explanation(query, doc)
                )
                retrieval_results.append(result)

            # Filter by minimum score
            if min_score > 0:
                retrieval_results = [r for r in retrieval_results if r.score >= min_score]

            logger.info(f"✅ Retrieved {len(retrieval_results)} relevant documents")

            return retrieval_results

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return []

    def _generate_relevance_explanation(self, query: str, document: Document) -> str:
        """
        Generate a brief explanation of why this document is relevant.

        Args:
            query: Original search query
            document: Retrieved document

        Returns:
            Explanation string
        """
        # Simple keyword matching - in practice, could use more sophisticated methods
        query_words = set(query.lower().split())
        doc_words = set(document.page_content.lower().split())

        matching_words = query_words.intersection(doc_words)
        if matching_words:
            return f"Contains keywords: {', '.join(list(matching_words)[:3])}"
        else:
            return "Semantic similarity match"

    def retrieve_with_reranking(
        self,
        query: str,
        k: int = 5,
        rerank_top_k: int = 20
    ) -> List[RetrievalResult]:
        """
        Retrieve documents with re-ranking for better relevance.

        Args:
            query: Search query
            k: Final number of documents to return
            rerank_top_k: Number of documents to retrieve initially for re-ranking

        Returns:
            Re-ranked list of RetrievalResult objects
        """
        # First, retrieve more documents than needed
        initial_results = self.retrieve(
            query=query,
            k=rerank_top_k,
            use_hybrid=True
        )

        if len(initial_results) <= k:
            return initial_results

        # Simple re-ranking based on content length and keyword matches
        # In practice, you might use a cross-encoder or other re-ranking model
        reranked_results = []
        for result in initial_results:
            # Boost score based on content quality indicators
            boost = 0.0

            # Prefer longer, more informative content
            content_length = len(result.document.page_content)
            if content_length > 500:
                boost += 0.1

            # Prefer content with categories/metadata
            if result.document.metadata.get("categories"):
                boost += 0.05

            # Prefer recent content (if timestamp available)
            if "timestamp" in result.document.metadata:
                boost += 0.05

            result.score = min(result.score + boost, 1.0)
            reranked_results.append(result)

        # Sort by new scores and return top k
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results[:k]

    def search_by_metadata(
        self,
        filters: Dict[str, Any],
        k: int = 10
    ) -> List[Document]:
        """
        Search documents by metadata filters.

        Args:
            filters: Metadata filters (e.g., {"source": "book", "categories": "ai"})
            k: Number of documents to return

        Returns:
            List of matching documents
        """
        try:
            # Use ChromaDB's metadata filtering
            results = self.vectorstore.similarity_search(
                query="",  # Empty query for metadata-only search
                k=k,
                filter=filters
            )
            logger.info(f"🔍 Found {len(results)} documents matching metadata filters")
            return results
        except Exception as e:
            logger.error(f"❌ Metadata search failed: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()

            # Get sample documents for analysis
            if count > 0:
                sample_docs = collection.get(limit=100)
                sources = set()
                categories = set()

                for metadata in sample_docs.get('metadatas', []):
                    if metadata:
                        if 'source' in metadata:
                            sources.add(metadata['source'])
                        if 'categories' in metadata:
                            categories.add(metadata['categories'])

                return {
                    "total_documents": count,
                    "unique_sources": len(sources),
                    "unique_categories": len(categories),
                    "sources": list(sources),
                    "categories": list(categories)
                }
            else:
                return {"total_documents": 0, "unique_sources": 0, "unique_categories": 0}

        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clear any cached retrieval results."""
        logger.info("🧹 Clearing retriever cache")
        # In practice, you might clear cached embeddings or similarity scores
        pass


# Example usage and testing
if __name__ == "__main__":
    # Initialize retriever
    retriever = SmartDocumentRetriever()

    # Example search
    query = "What is machine learning?"
    results = retriever.retrieve(query, k=3)

    print(f"\n🔍 Search Results for: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.document.page_content[:200]}...")
        print(f"   Score: {result.score:.3f}")
        print(f"   Source: {result.source}")
        print(f"   Explanation: {result.relevance_explanation}")
        print()

    # Get collection statistics
    stats = retriever.get_collection_stats()
    print("📊 Collection Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
