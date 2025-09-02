#!/usr/bin/env python3
"""
Demo script for SmartDocumentRetriever

This script demonstrates the advanced document retrieval capabilities
of the Smart Second Brain system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentic.workflows.document_retriever import SmartDocumentRetriever, RetrievalResult
from langchain.schema import Document


def create_sample_documents():
    """Create sample documents for demonstration."""
    return [
        Document(
            page_content="""
            Machine learning is a subset of artificial intelligence that enables computers
            to learn without being explicitly programmed. It uses algorithms and statistical
            models to analyze and draw inferences from patterns in data. Machine learning
            algorithms build mathematical models based on training data to make predictions
            or decisions without being explicitly programmed to perform the task.
            """,
            metadata={
                "source": "ai_textbook",
                "categories": "artificial_intelligence, machine_learning",
                "author": "Dr. Sarah Chen",
                "chapter": "Introduction to ML"
            }
        ),
        Document(
            page_content="""
            Deep learning is part of machine learning that uses neural networks with multiple
            layers. These networks can learn complex patterns and representations from large
            datasets. Deep learning has revolutionized fields like computer vision, natural
            language processing, and speech recognition through architectures like convolutional
            neural networks (CNNs) and transformers.
            """,
            metadata={
                "source": "neural_networks_guide",
                "categories": "deep_learning, neural_networks",
                "author": "Prof. Michael Torres",
                "chapter": "Advanced Neural Networks"
            }
        ),
        Document(
            page_content="""
            Natural language processing combines computational linguistics with statistical
            and machine learning models to give computers the ability to process and understand
            human language. Modern NLP techniques include tokenization, part-of-speech tagging,
            named entity recognition, and sentiment analysis. Large language models like GPT
            have dramatically improved the state-of-the-art in many NLP tasks.
            """,
            metadata={
                "source": "nlp_handbook",
                "categories": "natural_language_processing, ai",
                "author": "Dr. Lisa Wang",
                "chapter": "Modern NLP Techniques"
            }
        ),
        Document(
            page_content="""
            Vector databases store high-dimensional vectors and enable efficient similarity
            search. They are essential for modern AI applications including semantic search,
            recommendation systems, and retrieval-augmented generation. Vector databases use
            specialized indexing techniques like HNSW (Hierarchical Navigable Small World) to
            provide fast approximate nearest neighbor search at scale.
            """,
            metadata={
                "source": "database_technologies",
                "categories": "vector_databases, information_retrieval",
                "author": "Dr. James Liu",
                "chapter": "Advanced Database Systems"
            }
        ),
        Document(
            page_content="""
            Knowledge graphs represent information as nodes and edges, capturing relationships
            between entities. They are used in semantic search, question answering, and knowledge
            management systems. Knowledge graphs enable complex queries like "What papers did
            author X write about topic Y?" and support reasoning over interconnected data.
            """,
            metadata={
                "source": "semantic_web_journal",
                "categories": "knowledge_graphs, semantic_web",
                "author": "Dr. Robert Kim",
                "chapter": "Graph-Based Knowledge Representation"
            }
        )
    ]


def demonstrate_retrieval():
    """Demonstrate various retrieval capabilities."""

    print("🚀 SmartDocumentRetriever Demo")
    print("=" * 50)

    # Create temporary directory for demo
    import tempfile
    demo_dir = tempfile.mkdtemp(prefix="demo_chroma_")
    print(f"📁 Using temporary directory: {demo_dir}")

    try:
        # Initialize retriever
        print("\n🔧 Initializing SmartDocumentRetriever...")
        retriever = SmartDocumentRetriever(chromadb_dir=demo_dir)

        # Add sample documents
        print("📚 Adding sample documents...")
        sample_docs = create_sample_documents()
        retriever.add_documents(sample_docs)
        print(f"✅ Added {len(sample_docs)} documents")

        # Demonstrate different retrieval methods
        queries = [
            "machine learning algorithms",
            "neural network architectures",
            "natural language processing",
            "vector similarity search",
            "knowledge representation"
        ]

        print("\n🔍 Testing Basic Retrieval:")
        print("-" * 30)

        for query in queries:
            print(f"\nQuery: '{query}'")
            results = retriever.retrieve(query, k=2)

            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result.source}] {result.relevance_explanation}")
                print(".3f")
                print(f"     {result.document.page_content[:120]}...")

        # Demonstrate hybrid retrieval
        print("\n🔄 Testing Hybrid Retrieval (Semantic + Keyword):")
        print("-" * 50)

        query = "artificial intelligence techniques"
        hybrid_results = retriever.retrieve(query, k=3, use_hybrid=True)
        semantic_results = retriever.retrieve(query, k=3, use_hybrid=False)

        print(f"Query: '{query}'")
        print(f"Hybrid results: {len(hybrid_results)} documents")
        print(f"Semantic results: {len(semantic_results)} documents")

        # Demonstrate metadata filtering
        print("\n🏷️  Testing Metadata Filtering:")
        print("-" * 35)

        filters = {"categories": "machine_learning"}
        metadata_results = retriever.search_by_metadata(filters, k=5)

        print(f"Documents with category 'machine_learning': {len(metadata_results)}")
        for doc in metadata_results:
            print(f"  - {doc.metadata.get('source', 'unknown')}: {doc.metadata.get('chapter', 'N/A')}")

        # Demonstrate re-ranking
        print("\n📊 Testing Re-ranking:")
        print("-" * 25)

        query = "deep learning neural networks"
        reranked_results = retriever.retrieve_with_reranking(query, k=3, rerank_top_k=10)

        print(f"Query: '{query}'")
        print("Top 3 re-ranked results:")
        for i, result in enumerate(reranked_results, 1):
            print(f"  {i}. Score: {result.score:.3f} | {result.document.metadata.get('chapter', 'N/A')}")

        # Show collection statistics
        print("\n📈 Collection Statistics:")
        print("-" * 25)

        stats = retriever.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        import shutil
        if demo_dir and Path(demo_dir).exists():
            shutil.rmtree(demo_dir)
            print(f"\n🧹 Cleaned up demo directory: {demo_dir}")


def show_retrieval_result_structure():
    """Show the structure of RetrievalResult objects."""

    print("\n📋 RetrievalResult Structure:")
    print("-" * 30)

    # Example result
    result = RetrievalResult(
        document=Document(page_content="Sample content", metadata={"source": "test"}),
        score=0.85,
        source="test_source",
        relevance_explanation="Contains relevant keywords"
    )

    print("RetrievalResult {")
    print(f"  document: {type(result.document).__name__}")
    print(f"  score: {result.score} ({type(result.score).__name__})")
    print(f"  source: '{result.source}' ({type(result.source).__name__})")
    print(f"  relevance_explanation: '{result.relevance_explanation}'")
    print("}")

    print("\nDocument structure:")
    print(f"  page_content: Text content ({len(result.document.page_content)} chars)")
    print(f"  metadata: {result.document.metadata}")


if __name__ == "__main__":
    demonstrate_retrieval()
    show_retrieval_result_structure()

    print("\n🎉 Demo Complete!")
    print("The SmartDocumentRetriever provides:")
    print("  ✅ Semantic similarity search")
    print("  ✅ Hybrid retrieval (semantic + keyword)")
    print("  ✅ Metadata filtering")
    print("  ✅ Re-ranking and scoring")
    print("  ✅ Relevance explanations")
    print("  ✅ Collection statistics")
