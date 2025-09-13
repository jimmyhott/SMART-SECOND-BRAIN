"""
Master Graph Builder for Smart Second Brain LangGraph Workflows.

This module implements the core workflow orchestration system for the Smart Second Brain
project. It defines a comprehensive LangGraph workflow that handles both document
ingestion and knowledge querying through a series of interconnected nodes.

Key Workflows:
1. Document Ingestion: raw_document -> chunking -> embedding -> storage
2. Knowledge Query: user_input -> retrieval -> generation -> validation -> storage

Core Components:
- StateGraph: LangGraph workflow definition with conditional routing
- Node Methods: Individual processing steps for each workflow stage
- Checkpointing: Memory-based state persistence for conversation continuity
- Error Handling: Comprehensive error handling with fallback mechanisms

Architecture:
- Router Node: Determines workflow path based on query_type
- Ingestion Branch: Document processing and storage pipeline
- Query Branch: Knowledge retrieval and answer generation pipeline
- Human Review: Human-in-the-loop validation and feedback
- Validated Storage: Persistent storage of approved answers

Dependencies:
- LangGraph: Workflow orchestration and state management
- LangChain: Text processing, embeddings, and LLM integration
- ChromaDB: Vector database for document storage and retrieval
- Azure OpenAI: Embedding generation and language model interactions

Author: Smart Second Brain Team
Version: 0.1.0
"""

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import datetime
from ..core.knowledge_state import KnowledgeState
from ..core.conversation_memory import conversation_memory

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
    Main workflow orchestrator for Smart Second Brain knowledge processing.
    
    This class builds and manages the LangGraph workflow that handles both
    document ingestion and knowledge querying. It encapsulates all node
    definitions, workflow routing logic, and state management.
    
    The workflow supports two main paths:
    1. Document Ingestion: Processes new documents into the knowledge base
    2. Knowledge Query: Retrieves and generates answers from stored knowledge
    
    Key Features:
    - Conditional workflow routing based on query type
    - Comprehensive error handling and fallback mechanisms
    - Human-in-the-loop review and validation
    - Persistent state management with checkpointing
    - Flexible integration with various LLM and embedding providers
    
    Attributes:
        llm: Language model for text generation and reasoning
        embedding_model: Model for generating text embeddings
        retriever: Document retriever for similarity search
        vectorstore: Vector database for document storage
        chromadb_dir: Directory for ChromaDB persistence
        checkpointer: Memory-based state persistence system
    """

    def __init__(self, llm=None, embedding_model=None, retriever=None, vectorstore=None, chromadb_dir=None):
        """
        Initialize the MasterGraphBuilder with required dependencies.
        
        Args:
            llm: Language model client (e.g., ChatOpenAI, AzureChatOpenAI)
            embedding_model: Text embedding model (e.g., OpenAIEmbeddings, AzureOpenAIEmbeddings)
            retriever: Document retriever for similarity search (optional)
            vectorstore: Vector database client (optional, will be created if not provided)
            chromadb_dir: ChromaDB persistence directory (defaults to "./chroma_db")
            
        Note:
            At minimum, either llm or embedding_model should be provided for
            basic functionality. For full RAG capabilities, both are required.
        """
        self.llm = llm
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.chromadb_dir = chromadb_dir or "./chroma_db"
        
        # Initialize memory-based checkpointing for conversation continuity
        self.checkpointer = MemorySaver()

    # =============================================================================
    # WORKFLOW NODE METHODS
    # =============================================================================

    def input_router(self, state: KnowledgeState):
        """
        Route incoming requests to appropriate workflow branches.
        
        This node acts as the entry point for all workflows and determines
        the processing path based on the query_type field in the state.
        
        Args:
            state: KnowledgeState object containing workflow parameters
            
        Returns:
            KnowledgeState: Updated state object (unchanged for valid routes)
            
        Workflow Routes:
            - "ingest": Document ingestion workflow
            - "query": Knowledge query workflow
            - Unknown: Error state with logging
        """
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
        Split raw document text into manageable chunks for embedding.
        
        This node processes the raw document text and splits it into
        overlapping chunks that are optimal for embedding generation.
        Uses RecursiveCharacterTextSplitter for intelligent text segmentation.
        
        Args:
            state: KnowledgeState object containing raw_document
            
        Returns:
            KnowledgeState: Updated state with chunks array
            
        Processing:
            - Splits text into 500-character chunks with 50-character overlap
            - Maintains semantic coherence within chunks
            - Logs chunking statistics for monitoring
            - Handles missing document gracefully
        """
        if state.raw_document:
            # Use RecursiveCharacterTextSplitter for intelligent text segmentation
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,     # Maximum characters per chunk
                chunk_overlap=50    # Overlap between consecutive chunks for context
            )
            state.chunks = splitter.split_text(state.raw_document)
            state.logs = (state.logs or []) + [
                f"Chunked document into {len(state.chunks)} chunks"
            ]
        else:
            state.logs = (state.logs or []) + ["No raw_document found for chunking"]

        return state

    def embed_node(self, state: KnowledgeState):
        """
        Generate vector embeddings for text chunks.
        
        This node converts text chunks into high-dimensional vector
        representations that capture semantic meaning for similarity search.
        Includes fallback mechanisms for error handling and missing models.
        
        Args:
            state: KnowledgeState object containing chunks array
            
        Returns:
            KnowledgeState: Updated state with embeddings array
            
        Processing:
            - Generates embeddings using the configured embedding model
            - Falls back to placeholder embeddings if model fails
            - Logs embedding generation progress and errors
            - Handles missing chunks gracefully
        """
        # Use embedding model if available, otherwise fallback to placeholder
        if state.chunks:
            if self.embedding_model:
                try:
                    logger.info(f"🔤 Generating embeddings for {len(state.chunks)} chunks")
                    state.embeddings = self.embedding_model.embed_documents(state.chunks)
                    logger.info(f"✅ Generated {len(state.embeddings)} embeddings")
                except Exception as e:
                    logger.error(f"❌ Embedding generation failed: {e}")
                    # Fallback to placeholder embeddings for continued processing
                    state.embeddings = [[0.1, 0.2]] * len(state.chunks)
                    state.logs = (state.logs or []) + [f"Embedding error: {str(e)}"]
            else:
                logger.warning("⚠️ No embedding model provided, using placeholder embeddings")
                state.embeddings = [[0.1, 0.2]] * len(state.chunks)
        return state

    def store_node(self, state: KnowledgeState):
        """
        Store embeddings and chunks into ChromaDB vector database.
        
        This node persists the processed document chunks and their
        embeddings into the vector database for future retrieval.
        Includes comprehensive metadata for organization and traceability.
        
        Args:
            state: KnowledgeState object containing chunks, embeddings, and metadata
            
        Returns:
            KnowledgeState: Updated state with storage status
            
        Processing:
            - Creates ChromaDB collection if not exists
            - Stores text chunks with metadata (source, categories, chunk_id)
            - Handles metadata formatting for ChromaDB compatibility
            - Logs storage operations and errors
            - Updates state status upon completion
        """
        if state.embeddings and state.chunks:
            try:
                # Initialize vectorstore if not provided
                if not self.vectorstore:
                    self.vectorstore = Chroma(
                        collection_name="knowledge_base",
                        embedding_function=self.embedding_model,
                        persist_directory=self.chromadb_dir
                    )

                # Prepare metadata for each chunk
                # Note: ChromaDB doesn't accept lists in metadata, so we join categories
                metadatas = [
                    {
                        "source": state.source or "unknown",
                        "categories": ", ".join(state.categories) if state.categories else "general",
                        "chunk_id": i
                    }
                    for i in range(len(state.chunks))
                ]

                # Store chunks with metadata in ChromaDB
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
        Retrieve relevant documents from the vector database.
        
        This node performs similarity search to find the most relevant
        documents for a given user query. Supports multiple retrieval
        strategies with fallback mechanisms.
        
        Args:
            state: KnowledgeState object containing user_input
            
        Returns:
            KnowledgeState: Updated state with retrieved_docs array
            
        Retrieval Strategies:
            1. Dedicated retriever (if configured)
            2. Vectorstore similarity search (fallback)
            3. Empty results (if both unavailable)
            
        Processing:
            - Performs semantic search using user query
            - Retrieves top-k most relevant documents
            - Formats results for downstream processing
            - Handles retrieval errors gracefully
            - Logs retrieval operations and statistics
        """
        if not state.user_input:
            state.logs = (state.logs or []) + ["No user_input provided for retrieval"]
            state.retrieved_docs = []
            return state

        try:
            # Try different retrieval strategies in order of preference
            if self.retriever:
                logger.info(f"🔍 Retrieving docs for query: {state.user_input}")
                results = self.retriever.get_relevant_documents(state.user_input)
            elif self.vectorstore:
                logger.info(f"🔍 Using vectorstore.similarity_search for query: {state.user_input}")
                results = self.vectorstore.similarity_search(state.user_input, k=5)
            else:
                results = []

            # Format retrieved documents for downstream processing
            state.retrieved_docs = [{"content": r.page_content, "metadata": r.metadata} for r in results]
            state.logs = (state.logs or []) + [f"Retrieved {len(state.retrieved_docs)} docs"]

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            state.retrieved_docs = []
            state.logs = (state.logs or []) + [f"Retrieval error: {str(e)}"]

        return state

    def answer_gen_node(self, state: KnowledgeState):
        """
        Generate AI-powered answers using retrieved context and conversation history.
        
        This node implements the core RAG (Retrieval-Augmented Generation) step,
        combining retrieved documents with conversation context to generate
        grounded, relevant answers to user queries.
        
        Args:
            state: KnowledgeState object containing user_input, retrieved_docs, and messages
            
        Returns:
            KnowledgeState: Updated state with generated_answer and updated messages
            
        RAG Implementation:
            - Context: Retrieved document content
            - Conversation: User-assistant message history
            - Query: Current user question
            - Instructions: Clear guidelines for answer generation
            
        Processing:
            - Builds context from retrieved documents
            - Formats conversation history
            - Uses ChatPromptTemplate for structured prompting
            - Calls language model with full context
            - Updates conversation history
            - Logs generation success/failure
        """
        if not self.llm or not state.user_input:
            state.logs = (state.logs or []) + ["⚠️ No LLM or user query provided"]
            return state

        try:
            # --- Build retrieved context ---
            context = "\n\n".join(
                [doc["content"] for doc in (state.retrieved_docs or [])]
            ) or "No relevant documents were retrieved."

            # --- Format conversation history ---
            conversation = "\n".join(
                [f"{m['role']}: {m['content']}" for m in (state.messages or [])]
            )

            # --- Define enhanced RAG prompt template with conversation context ---
            template = """
            You are a helpful knowledge assistant with access to conversation history.

            CONVERSATION HISTORY:
            {conversation}

            CURRENT USER QUESTION:
            {query}

            RETRIEVED KNOWLEDGE BASE CONTEXT:
            {context}

            INSTRUCTIONS:
            - Use the conversation history to understand context and references
            - Base your answer primarily on the retrieved knowledge base context
            - If the knowledge base context is insufficient, say "I don't know based on available knowledge."
            - Consider previous questions and answers to provide better context
            - Keep the answer clear, concise, and contextually relevant
            - If the user refers to something from earlier in the conversation, acknowledge it
            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm

            # --- Call language model with full context ---
            response = chain.invoke({
                "conversation": conversation,
                "query": state.user_input,
                "context": context
            })

            # --- Update state with generated answer ---
            state.generated_answer = response.content
            
            # Initialize messages array if not exists
            if not state.messages:
                state.messages = []
                
            # Add current user input and assistant response to conversation history
            # Note: Conversation memory is now managed by custom ConversationMemoryManager
            state.messages.append({"role": "user", "content": state.user_input})
            state.messages.append({"role": "assistant", "content": state.generated_answer})
            state.logs = (state.logs or []) + ["✅ Generated contextual answer with conversation history"]

        except Exception as e:
            state.generated_answer = None
            state.logs = (state.logs or []) + [f"❌ Answer generation failed: {e}"]

        return state

    def human_review_node(self, state: KnowledgeState):
        """
        Handle human-in-the-loop review and validation of AI-generated answers.
        
        This node manages the human review process, supporting various
        feedback types including approval, rejection, and manual editing.
        Implements auto-approval for non-interactive scenarios.
        
        Args:
            state: KnowledgeState object containing generated_answer and optional feedback
            
        Returns:
            KnowledgeState: Updated state with final_answer and feedback status
            
        Feedback Types:
            - "approved": Answer meets quality standards
            - "rejected": Answer doesn't meet requirements
            - "edited": Answer was manually modified
            - None: Auto-approval (default behavior)
            
        Processing:
            - Checks for explicit human feedback
            - Implements auto-approval for batch processing
            - Handles rejection and editing scenarios
            - Updates final_answer based on feedback
            - Logs review decisions and actions
        """
        try:
            # Check for explicit human feedback
            feedback = getattr(state, "human_feedback", None)

            if not feedback:
                # Auto-approval path for batch or non-interactive runs
                feedback = "approved"
                state.final_answer = state.generated_answer
                state.logs = (state.logs or []) + ["✅ Auto-approved answer"]

            elif feedback == "rejected":
                # Handle rejection - no final answer
                state.final_answer = None
                state.logs = (state.logs or []) + ["❌ Answer was rejected by human"]

            elif feedback == "edited":
                # Handle manual editing
                if hasattr(state, "edited_answer") and state.edited_answer:
                    state.final_answer = state.edited_answer
                    state.logs = (state.logs or []) + ["✏️ Human edited the answer"]
                else:
                    state.final_answer = None
                    state.logs = (state.logs or []) + ["⚠️ Edit requested but no edited_answer provided"]

            else:
                # Unknown feedback type - default to generated answer
                state.final_answer = state.generated_answer
                state.logs = (state.logs or []) + [f"⚠️ Unknown feedback '{feedback}', defaulting to generated answer"]

            # Save feedback for traceability
            state.human_feedback = feedback

        except Exception as e:
            # Error handling - default to generated answer
            state.final_answer = state.generated_answer
            state.human_feedback = "error"
            state.logs = (state.logs or []) + [f"❌ Human review failed: {e}"]

        return state

    def validated_store_node(self, state: KnowledgeState):
        """
        Intelligent storage routing based on knowledge type and human feedback.
        
        This node implements smart storage decisions:
        - Conversational content: Stored in checkpoint memory only
        - Reusable/verified knowledge: Stored in both checkpoint and vector DB
        
        Args:
            state: KnowledgeState object containing final_answer, feedback, and knowledge_type
            
        Returns:
            KnowledgeState: Updated state with storage status
            
        Storage Logic:
            - Default: Store in checkpoint memory (conversational)
            - knowledge_type='reusable' or 'verified': Also store in vector DB
            - Rejected content: Not stored anywhere
        """
        try:
            # Check for answer to validate
            if not state.final_answer:
                state.logs = (state.logs or []) + ["⚠️ No final_answer to validate, skipping"]
                return state

            # Check feedback status
            if state.human_feedback not in ("approved", "edited"):
                state.logs = (state.logs or []) + [f"ℹ️ Skipping validation because feedback = {state.human_feedback}"]
                return state

            # Determine knowledge type (default to conversational)
            knowledge_type = getattr(state, "knowledge_type", "conversational")
            
            # Always store in checkpoint memory for conversation continuity
            state.status = "validated"
            state.logs = (state.logs or []) + ["✅ Conversation history managed by LangGraph checkpoint memory (thread-isolated)"]
            
            # Check if content should also be stored in vector DB
            if knowledge_type in ("reusable", "verified"):
                try:
                    # Create a document for vector storage
                    from langchain.schema import Document
                    
                    # Prepare metadata for vector storage
                    metadata = {
                        "source": f"human_approved_{state.query_type}",
                        "thread_id": getattr(state, "thread_id", "unknown"),
                        "knowledge_type": knowledge_type,
                        "human_feedback": state.human_feedback,
                        "timestamp": datetime.now().isoformat(),
                        "user_input": state.user_input or "unknown_query"
                    }
                    
                    # Add any existing metadata
                    if state.metadata:
                        metadata.update(state.metadata)
                    
                    # Create document for vector storage
                    doc = Document(
                        page_content=state.final_answer,
                        metadata=metadata
                    )
                    
                    # Store in vector database
                    if self.vectorstore:
                        self.vectorstore.add_documents([doc])
                        state.logs = (state.logs or []) + [f"✅ {knowledge_type.title()} knowledge stored in vector database"]
                        logger.info(f"📚 Stored {knowledge_type} knowledge in vector DB: {state.final_answer[:100]}...")
                    else:
                        state.logs = (state.logs or []) + ["⚠️ Vector store not available, skipping vector storage"]
                        
                except Exception as e:
                    state.logs = (state.logs or []) + [f"❌ Failed to store in vector DB: {e}"]
                    logger.error(f"Failed to store knowledge in vector DB: {e}")
            else:
                # Conversational content - checkpoint memory only
                state.logs = (state.logs or []) + ["💬 Conversational content stored in checkpoint memory only"]

        except Exception as e:
            state.status = "error"
            state.logs = (state.logs or []) + [f"❌ Validation failed: {e}"]

        return state

    # =============================================================================
    # GRAPH COMPILATION
    # =============================================================================

    def build(self) -> CompiledStateGraph:
        """
        Build and compile the complete LangGraph workflow.
        
        This method constructs the workflow graph by defining nodes,
        setting up conditional routing, and establishing the flow
        between different processing stages.
        
        Returns:
            CompiledStateGraph: Compiled and checkpointed workflow graph
            
        Graph Structure:
            Entry Point: router
            Conditional Routing: Based on query_type
            Ingestion Branch: router -> chunk -> embed -> store -> END
            Query Branch: router -> retriever -> answer -> review -> validated_store -> END
            
        Features:
            - Conditional edges for workflow routing
            - Checkpointing for state persistence
            - Error handling and logging throughout
            - Human-in-the-loop integration
        """
        # Create the base state graph
        graph = StateGraph(KnowledgeState)

        # =============================================================================
        # NODE DEFINITIONS
        # =============================================================================
        
        # Add all workflow nodes
        graph.add_node("router", self.input_router)           # Entry point and routing
        graph.add_node("chunk", self.chunk_doc_node)          # Document chunking
        graph.add_node("embed", self.embed_node)              # Embedding generation
        graph.add_node("store", self.store_node)              # Document storage
        graph.add_node("retriever", self.retriever_node)      # Document retrieval
        graph.add_node("answer", self.answer_gen_node)        # Answer generation
        graph.add_node("review", self.human_review_node)      # Human review
        graph.add_node("validated_store", self.validated_store_node)  # Validated storage

        # Set the entry point
        graph.set_entry_point("router")

        # =============================================================================
        # CONDITIONAL ROUTING
        # =============================================================================
        
        # Define routing logic based on query_type
        def route_condition(state):
            if hasattr(state, 'query_type'):
                if state.query_type == "ingest":
                    return "ingest"
                elif state.query_type == "query":
                    return "query"
                else:
                    return "__end__"
            return "query"
        
        # Add conditional edges from router
        graph.add_conditional_edges("router", route_condition, {
            "ingest": "chunk",      # Route to document ingestion
            "query": "retriever",   # Route to knowledge query
            "__end__": END,         # End workflow for invalid types
        })

        # =============================================================================
        # INGESTION BRANCH
        # =============================================================================
        
        # Document processing pipeline
        graph.add_edge("chunk", "embed")      # Chunk -> Embed
        graph.add_edge("embed", "store")     # Embed -> Store
        graph.add_edge("store", END)         # Store -> End

        # =============================================================================
        # QUERY BRANCH
        # =============================================================================
        
        # Knowledge query pipeline
        graph.add_edge("retriever", "answer")           # Retrieve -> Generate
        graph.add_edge("answer", "review")             # Generate -> Review
        graph.add_edge("review", "validated_store")    # Review -> Store
        graph.add_edge("validated_store", END)         # Store -> End

        # Compile the graph with checkpointing
        return graph.compile(checkpointer=self.checkpointer)