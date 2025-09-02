from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import datetime
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
        """
        Retrieves relevant documents from Chroma based on user query.
        Falls back to dummy text if retriever/vectorstore not available.
        """
        if not state.user_input:
            state.logs = (state.logs or []) + ["No user_input provided for retrieval"]
            state.retrieved_docs = []
            return state

        try:
            if self.retriever:
                logger.info(f"🔍 Retrieving docs for query: {state.user_input}")
                results = self.retriever.get_relevant_documents(state.user_input)
            elif self.vectorstore:
                logger.info(f"🔍 Using vectorstore.similarity_search for query: {state.user_input}")
                results = self.vectorstore.similarity_search(state.user_input, k=5)
            else:
                results = []

            state.retrieved_docs = [{"content": r.page_content, "metadata": r.metadata} for r in results]
            state.logs = (state.logs or []) + [f"Retrieved {len(state.retrieved_docs)} docs"]

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            state.retrieved_docs = []
            state.logs = (state.logs or []) + [f"Retrieval error: {str(e)}"]

        return state


    def answer_gen_node(self, state: KnowledgeState):
        """
        Generates an answer using retrieved docs + conversation history + user query.
        Implements a proper RAG step with fallback.
        """
        if not self.llm or not state.user_input:
            state.logs = (state.logs or []) + ["⚠️ No LLM or user query provided"]
            return state

        try:
            # --- Retrieved context ---
            context = "\n\n".join(
                [doc["content"] for doc in (state.retrieved_docs or [])]
            ) or "No relevant documents were retrieved."

            # --- Conversation history ---
            conversation = "\n".join(
                [f"{m['role']}: {m['content']}" for m in (state.messages or [])]
            )

            # --- Prompt template ---
            template = """
            You are a helpful knowledge assistant.

            Conversation so far:
            {conversation}

            User question:
            {query}

            Retrieved context:
            {context}

            Instructions:
            - Base your answer primarily on the retrieved context.
            - If the context is empty or insufficient, say "I don’t know based on available knowledge."
            - Keep the answer clear and concise.
            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm

            # --- Call LLM ---
            response = chain.invoke({
                "conversation": conversation,
                "query": state.user_input,
                "context": context
            })

            # --- Update state ---
            state.generated_answer = response.content
            if not state.messages:
                state.messages = []
            state.messages.append({"role": "user", "content": state.user_input})
            state.messages.append({"role": "assistant", "content": state.generated_answer})
            state.logs = (state.logs or []) + ["✅ Generated grounded answer"]

        except Exception as e:
            state.generated_answer = None
            state.logs = (state.logs or []) + [f"❌ Answer generation failed: {e}"]

        return state


    def human_review_node(self, state: KnowledgeState):
        """
        Handles human-in-the-loop review of the AI's answer.
        In a real app, this could be connected to a UI or workflow tool.
        For now, supports auto-approval with the option to simulate rejection/edits.
        """
        try:
            # By default, auto-approve (for batch or non-interactive runs)
            feedback = getattr(state, "human_feedback", None)

            if not feedback:
                # Auto-approve path
                feedback = "approved"
                state.final_answer = state.generated_answer
                state.logs = (state.logs or []) + ["✅ Auto-approved answer"]

            elif feedback == "rejected":
                # If user rejected, final_answer is None
                state.final_answer = None
                state.logs = (state.logs or []) + ["❌ Answer was rejected by human"]

            elif feedback == "edited":
                # If user edited, expect state.edited_answer to exist
                if hasattr(state, "edited_answer") and state.edited_answer:
                    state.final_answer = state.edited_answer
                    state.logs = (state.logs or []) + ["✏️ Human edited the answer"]
                else:
                    state.final_answer = None
                    state.logs = (state.logs or []) + ["⚠️ Edit requested but no edited_answer provided"]

            else:
                # Unknown feedback type
                state.final_answer = state.generated_answer
                state.logs = (state.logs or []) + [f"⚠️ Unknown feedback '{feedback}', defaulting to generated answer"]

            # Save feedback
            state.human_feedback = feedback

        except Exception as e:
            state.final_answer = state.generated_answer
            state.human_feedback = "error"
            state.logs = (state.logs or []) + [f"❌ Human review failed: {e}"]

        return state

    
    def validated_store_node(self, state: KnowledgeState):
        """
        Stores human-validated answers back into Chroma as new knowledge.
        Only stores if feedback is 'approved' or 'edited'.
        """
        try:
            if not self.vectorstore:
                state.logs = (state.logs or []) + ["⚠️ No vectorstore available, skipping validated store"]
                return state

            if not state.final_answer:
                state.logs = (state.logs or []) + ["⚠️ No final_answer to store, skipping"]
                return state

            if state.human_feedback not in ("approved", "edited"):
                state.logs = (state.logs or []) + [f"ℹ️ Skipping store because feedback = {state.human_feedback}"]
                return state

            # Add metadata for traceability
            metadata = {
                "source": "assistant_validated",
                "categories": ", ".join(state.categories) if state.categories else "general",
                "feedback": state.human_feedback,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }

            self.vectorstore.add_texts(
                texts=[state.final_answer],
                metadatas=[metadata]
            )

            state.status = "validated"
            state.logs = (state.logs or []) + ["✅ Stored validated answer in Chroma"]

        except Exception as e:
            state.status = "error"
            state.logs = (state.logs or []) + [f"❌ Validated store failed: {e}"]

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