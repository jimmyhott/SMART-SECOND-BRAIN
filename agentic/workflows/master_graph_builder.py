from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from ..core.knowledge_state import KnowledgeState

# Import centralized logging
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from shared.utils.logging_config import setup_agentic_logging

# Set up agentic logging
logger = setup_agentic_logging()


class MasterGraphBuilder:
    """
    Class to build the Smart Second Brain master graph.
    Encapsulates node definitions, dependencies, and graph wiring.
    """

    def __init__(self, llm=None, embedding_model=None, retriever=None, vectorstore=None, chromadb_dir=None):
        """
        Initialize dependencies.
        :param llm: LLM client (e.g., ChatOpenAI)
        :param embedding_model: Text embedding model (e.g., OpenAIEmbeddings)
        :param retriever: Document retriever (optional)
        :param vectorstore: Vector DB client (optional)
        :param chromadb_dir: ChromaDB persistence directory (optional, defaults to "./chroma_db")
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.chromadb_dir = chromadb_dir or "./chroma_db"
        self.checkpointer = MemorySaver()


    # --- Node Methods ---
    def input_router(self, state: KnowledgeState):
        if state.query_type == "ingest":
            return state
        elif state.query_type == "query":
            return state
        else:
            state.status = "error"
            state.logs = (state.logs or []) + ["Unknown query_type"]
            return state

    def chunk_doc_node(self, state: KnowledgeState):
        """
        Splits raw text into overlapping chunks for embedding.
        Uses RecursiveCharacterTextSplitter for better balance and robustness.
        """
        if state.raw_document:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,     # max characters per chunk
                chunk_overlap=50    # overlap between chunks
            )
            state.chunks = splitter.split_text(state.raw_document)
            state.logs = (state.logs or []) + [
                f"Chunked document into {len(state.chunks)} chunks"
            ]
        else:
            state.logs = (state.logs or []) + ["No raw_document found for chunking"]

        return state


    def embed_node(self, state: KnowledgeState):
        # Use embedding model if available, otherwise fallback to placeholder
        if state.chunks:
            if self.embedding_model:
                try:
                    logger.info(f"🔤 Generating embeddings for {len(state.chunks)} chunks")
                    state.embeddings = self.embedding_model.embed_documents(state.chunks)
                    logger.info(f"✅ Generated {len(state.embeddings)} embeddings")
                except Exception as e:
                    logger.error(f"❌ Embedding generation failed: {e}")
                    # Fallback to placeholder embeddings
                    state.embeddings = [[0.1, 0.2]] * len(state.chunks)
                    state.logs = (state.logs or []) + [f"Embedding error: {str(e)}"]
            else:
                logger.warning("⚠️ No embedding model provided, using placeholder embeddings")
                state.embeddings = [[0.1, 0.2]] * len(state.chunks)
        return state

    def store_node(self, state: KnowledgeState):
        """
        Stores embeddings + chunks into ChromaDB with metadata including multiple categories.
        """
        if state.embeddings and state.chunks:
            try:
                if not self.vectorstore:
                    self.vectorstore = Chroma(
                        collection_name="knowledge_base",
                        embedding_function=self.embedding_model,
                        persist_directory=self.chromadb_dir
                    )

                # Store categories as a string (ChromaDB doesn't accept lists in metadata)
                metadatas = [
                    {
                        "source": state.source or "unknown",
                        "categories": ", ".join(state.categories) if state.categories else "general",
                        "chunk_id": i
                    }
                    for i in range(len(state.chunks))
                ]

                self.vectorstore.add_texts(
                    texts=state.chunks,
                    metadatas=metadatas
                )

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
        if self.retriever and state.user_input:
            results = self.retriever.get_relevant_documents(state.user_input)
            state.retrieved_docs = [{"content": r.page_content} for r in results]
        else:
            state.retrieved_docs = [{"content": "Dummy retrieved note"}]
        return state

    def answer_gen_node(self, state: KnowledgeState):
        if self.llm and state.user_input:
            context = "\n".join([doc["content"] for doc in state.retrieved_docs or []])
            prompt = [
                {"role": "system", "content": "You are a helpful knowledge assistant."},
                {"role": "user", "content": f"Query: {state.user_input}\n\nContext:\n{context}"}
            ]
            response = self.llm.invoke(prompt)
            state.generated_answer = response.content
            state.messages.append({"role": "ai", "content": state.generated_answer})
        return state

    def human_review_node(self, state: KnowledgeState):
        # Stub: in real app, integrate with UI or approval workflow
        state.human_feedback = "approved"
        state.final_answer = state.generated_answer
        return state

    def validated_store_node(self, state: KnowledgeState):
        # Store validated answer (optional)
        if self.vectorstore and state.final_answer:
            self.vectorstore.add_texts([state.final_answer])
        state.status = "validated"
        return state

    # --- Graph Compilation ---
    def build(self) -> CompiledStateGraph:
        """Build and compile the master graph."""
        graph = StateGraph(KnowledgeState)

        # Nodes
        graph.add_node("router", self.input_router)
        graph.add_node("chunk", self.chunk_doc_node)
        graph.add_node("embed", self.embed_node)
        graph.add_node("store", self.store_node)
        graph.add_node("retriever", self.retriever_node)
        graph.add_node("answer", self.answer_gen_node)
        graph.add_node("review", self.human_review_node)
        graph.add_node("validated_store", self.validated_store_node)

        # Entry point
        graph.set_entry_point("router")

        # Edges
        def route_condition(state):
            if hasattr(state, 'query_type'):
                if state.query_type == "ingest":
                    return "ingest"
                elif state.query_type == "query":
                    return "query"
                else:
                    return "__end__"
            return "query"
        
        graph.add_conditional_edges("router", route_condition, {
            "ingest": "chunk",
            "query": "retriever",
            "__end__": END,
        })

        # Ingest branch
        graph.add_edge("chunk", "embed")
        graph.add_edge("embed", "store")
        graph.add_edge("store", END)

        # Query branch
        graph.add_edge("retriever", "answer")
        graph.add_edge("answer", "review")
        graph.add_edge("review", "validated_store")
        graph.add_edge("validated_store", END)

        return graph.compile(checkpointer=self.checkpointer)