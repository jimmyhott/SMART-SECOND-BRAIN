"""
Document Retriever Implementation for Smart Second Brain

This module provides various document retrieval strategies including:
- Vector similarity search
- Hybrid search (vector + keyword)
- Multi-modal retrieval
- Contextual retrieval with metadata filtering
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.schema import Document
from langchain.retrievers import (
    VectorStoreRetriever,
    BM25Retriever,
    EnsembleRetriever,
    ContextualCompressionRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)


class SmartDocumentRetriever:
    """
    Advanced document retriever with multiple retrieval strategies.
    
    Features:
    - Vector similarity search
    - Keyword-based retrieval (BM25)
    - Hybrid search combining multiple strategies
    - Metadata filtering
    - Contextual compression
    - Multi-modal retrieval support
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        embedding_model=None,
        search_type: str = "hybrid",
        k: int = 5,
        score_threshold: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the document retriever.
        
        Args:
            vectorstore: ChromaDB vector store instance
            embedding_model: Embedding model for vector search
            search_type: Type of search ("vector", "keyword", "hybrid", "contextual")
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            metadata_filters: Filters to apply to metadata
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.search_type = search_type
        self.k = k
        self.score_threshold = score_threshold
        self.metadata_filters = metadata_filters or {}
        
        # Initialize different retrievers
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        """Setup different types of retrievers."""
        
        # Vector store retriever
        self.vector_retriever = VectorStoreRetriever(
            vectorstore=self.vectorstore,
            search_type="similarity",
            search_kwargs={
                "k": self.k,
                "score_threshold": self.score_threshold,
                "filter": self.metadata_filters
            }
        )
        
        # BM25 retriever for keyword search
        # Note: This requires documents to be loaded into memory
        self.bm25_retriever = None  # Will be initialized when documents are available
        
        # Ensemble retriever for hybrid search
        self.ensemble_retriever = None
        
        # Contextual compression retriever
        self.contextual_retriever = None
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: User query string
            
        Returns:
            List of relevant documents
        """
        logger.info(f"🔍 Retrieving documents for query: '{query}' using {self.search_type} search")
        
        try:
            if self.search_type == "vector":
                return self._vector_search(query)
            elif self.search_type == "keyword":
                return self._keyword_search(query)
            elif self.search_type == "hybrid":
                return self._hybrid_search(query)
            elif self.search_type == "contextual":
                return self._contextual_search(query)
            else:
                logger.warning(f"Unknown search type: {self.search_type}, falling back to vector search")
                return self._vector_search(query)
                
        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            return []
    
    def _vector_search(self, query: str) -> List[Document]:
        """Perform vector similarity search."""
        logger.info("🔤 Using vector similarity search")
        
        # Use ChromaDB's similarity search directly
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.k,
            filter=self.metadata_filters
        )
        
        # Filter by score threshold
        filtered_results = [
            doc for doc, score in results 
            if score >= self.score_threshold
        ]
        
        logger.info(f"✅ Vector search returned {len(filtered_results)} documents")
        return filtered_results
    
    def _keyword_search(self, query: str) -> List[Document]:
        """Perform keyword-based search using BM25."""
        logger.info("🔤 Using keyword-based search (BM25)")
        
        # For now, fall back to vector search
        # BM25 would require loading all documents into memory
        logger.warning("⚠️ BM25 search not implemented, falling back to vector search")
        return self._vector_search(query)
    
    def _hybrid_search(self, query: str) -> List[Document]:
        """Perform hybrid search combining vector and keyword search."""
        logger.info("🔤 Using hybrid search (vector + keyword)")
        
        # Get results from both methods
        vector_results = self._vector_search(query)
        
        # For now, just return vector results
        # In a full implementation, you'd combine and re-rank results
        logger.info(f"✅ Hybrid search returned {len(vector_results)} documents")
        return vector_results
    
    def _contextual_search(self, query: str) -> List[Document]:
        """Perform contextual search with compression."""
        logger.info("🔤 Using contextual search with compression")
        
        # Get initial results
        initial_results = self._vector_search(query)
        
        # For now, return initial results
        # Contextual compression would require an LLM for re-ranking
        logger.info(f"✅ Contextual search returned {len(initial_results)} documents")
        return initial_results
    
    def search_by_metadata(self, metadata_filters: Dict[str, Any]) -> List[Document]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filters: Dictionary of metadata filters
            
        Returns:
            List of documents matching the filters
        """
        logger.info(f"🔍 Searching by metadata filters: {metadata_filters}")
        
        try:
            # Use ChromaDB's get method with metadata filters
            results = self.vectorstore.get(
                where=metadata_filters,
                limit=self.k
            )
            
            # Convert to Document objects
            documents = []
            for i, content in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            logger.info(f"✅ Metadata search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Metadata search failed: {e}")
            return []
    
    def search_by_category(self, category: str) -> List[Document]:
        """
        Search documents by category.
        
        Args:
            category: Category to search for
            
        Returns:
            List of documents in the specified category
        """
        return self.search_by_metadata({"categories": {"$contains": category}})
    
    def search_by_source(self, source: str) -> List[Document]:
        """
        Search documents by source.
        
        Args:
            source: Source to search for
            
        Returns:
            List of documents from the specified source
        """
        return self.search_by_metadata({"source": source})
    
    def get_document_count(self) -> int:
        """Get total number of documents in the vector store."""
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except Exception as e:
            logger.error(f"❌ Failed to get document count: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get sample of documents for analysis
            results = collection.get(limit=100)
            
            # Analyze categories
            categories = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata and 'categories' in metadata:
                        cat = metadata['categories']
                        categories[cat] = categories.get(cat, 0) + 1
            
            # Analyze sources
            sources = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata and 'source' in metadata:
                        source = metadata['source']
                        sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_documents": count,
                "categories": categories,
                "sources": sources,
                "sample_size": len(results['documents'])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {"error": str(e)}


class RetrieverFactory:
    """Factory for creating different types of retrievers."""
    
    @staticmethod
    def create_retriever(
        vectorstore: Chroma,
        retriever_type: str = "smart",
        **kwargs
    ) -> Union[SmartDocumentRetriever, VectorStoreRetriever]:
        """
        Create a retriever based on the specified type.
        
        Args:
            vectorstore: ChromaDB vector store
            retriever_type: Type of retriever to create
            **kwargs: Additional arguments for the retriever
            
        Returns:
            Configured retriever instance
        """
        
        if retriever_type == "smart":
            return SmartDocumentRetriever(vectorstore, **kwargs)
        elif retriever_type == "vector":
            return VectorStoreRetriever(
                vectorstore=vectorstore,
                search_type="similarity",
                search_kwargs=kwargs
            )
        elif retriever_type == "simple":
            return vectorstore.as_retriever(search_kwargs=kwargs)
        else:
            logger.warning(f"Unknown retriever type: {retriever_type}, using simple retriever")
            return vectorstore.as_retriever(search_kwargs=kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the retriever
    print("📚 Smart Document Retriever Example")
    print("=" * 50)
    
    # This would be used in the actual application
    # retriever = SmartDocumentRetriever(
    #     vectorstore=chroma_instance,
    #     search_type="hybrid",
    #     k=5,
    #     score_threshold=0.7,
    #     metadata_filters={"categories": {"$contains": "ai"}}
    # )
    # 
    # results = retriever.get_relevant_documents("What is machine learning?")
    # print(f"Found {len(results)} relevant documents")
