#!/usr/bin/env python3
"""
Smart Second Brain - Streamlit Frontend
A modern web interface for AI-powered knowledge management
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import requests
from streamlit_chat import message

# Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/health",
    "ingest": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/ingest",
    "ingest_pdfs": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/ingest-pdfs",
    "query": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/query",
    "feedback": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/feedback",
    "feedback_status": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/feedback",
}

# Page configuration
st.set_page_config(
    page_title="Smart Second Brain",
    page_icon="second-brain.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with Bootstrap
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<style>
    .health-status {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #007bff;
    }
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .status-healthy { background: #28a745; }
    .status-error { background: #dc3545; }
    .status-warning { background: #ffc107; }
    .chat-message {
        padding: 0.75rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        max-width: 70%;
        word-wrap: break-word;
    }
    .user-message {
        background: #007bff;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .assistant-message {
        background: #f8f9fa;
        color: #495057;
        border: 1px solid #dee2e6;
        border-bottom-left-radius: 4px;
    }
    .upload-area {
        border: 2px dashed #007bff;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #0056b3;
        background: #e3f2fd;
    }
    .file-item {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_health' not in st.session_state:
    st.session_state.system_health = {
        "health": "unknown",
        "graph_initialized": False,
        "vectorstore_ready": False,
        "embedding_model_ready": False,
        "llm_ready": False,
        "last_check": None
    }
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_thread_id' not in st.session_state:
    st.session_state.current_thread_id = None
if 'new_thread_clicked' not in st.session_state:
    st.session_state.new_thread_clicked = False
if 'pending_feedback' not in st.session_state:
    st.session_state.pending_feedback = {}

# API Functions
def api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None, files: Optional[Dict] = None) -> Dict:
    """Make API request to backend"""
    try:
        if method == "GET":
            response = requests.get(endpoint, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(endpoint, files=files, data=data, timeout=60)
            else:
                response = requests.post(endpoint, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def check_system_health():
    """Check system health status"""
    try:
        result = api_request(API_ENDPOINTS["health"])
        
        if "error" not in result:
            st.session_state.system_health.update({
                "health": result.get("status", "unknown"),
                "graph_initialized": result.get("graph_initialized", False),
                "vectorstore_ready": result.get("vectorstore_ready", False),
                "embedding_model_ready": result.get("embedding_model_ready", False),
                "llm_ready": result.get("llm_ready", False),
                "last_check": datetime.now().strftime("%H:%M:%S")
            })
        else:
            st.session_state.system_health["health"] = "error"
            
    except Exception as e:
        st.session_state.system_health["health"] = "error"
        st.session_state.system_health["last_check"] = datetime.now().strftime("%H:%M:%S")


def ingest_multiple_pdfs(uploaded_files: List, source: str, categories: str = "", author: str = "", metadata: str = ""):
    """Ingest multiple PDF files into the knowledge base"""
    if not uploaded_files:
        st.error("Please upload PDF files")
        return False
    
    if not source.strip():
        st.error("Please provide a source for the PDF files")
        return False
    
    try:
        # Prepare files for upload
        files = []
        for file in uploaded_files:
            files.append(("files", (file.name, file.getvalue(), "application/pdf")))
        
        # Prepare form data
        data = {
            "source": source.strip(),
            "categories": categories.strip() if categories else "",
            "author": author.strip() if author else "",
            "metadata": metadata.strip() if metadata else ""
        }
        
        # Make API request
        result = api_request(API_ENDPOINTS["ingest_pdfs"], "POST", data, files)
        
        if "error" not in result:
            # Display success message with details
            total_files = result.get("total_files", 0)
            processed_files = result.get("processed_files", 0)
            failed_files = result.get("failed_files", 0)
            execution_time = result.get("execution_time", 0)
            
            if processed_files == total_files:
                st.success(f"✅ All {processed_files} PDF files processed successfully in {execution_time:.2f}s!")
            else:
                st.warning(f"⚠️ {processed_files}/{total_files} files processed successfully ({failed_files} failed) in {execution_time:.2f}s")
            
            # Show detailed results
            if result.get("results"):
                with st.expander("📊 Processing Details", expanded=False):
                    for file_result in result["results"]:
                        filename = file_result.get("filename", "unknown")
                        chunks = file_result.get("chunks_created", 0)
                        status = "✅" if file_result.get("success", False) else "❌"
                        st.write(f"{status} **{filename}**: {chunks} chunks created")
            
            return True
        else:
            st.error(f"Error processing PDFs: {result['error']}")
            return False
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def process_text_ingestion(content, source, categories=None, author=None, knowledge_type="reusable"):
    """Process text content ingestion using the /ingest endpoint"""
    try:
        # Prepare data for API request
        data = {
            "content": content,
            "source": source,
            "categories": categories,
            "author": author,
            "knowledge_type": knowledge_type
        }
        
        # Make API request
        result = api_request(API_ENDPOINTS["ingest"], "POST", data)
        
        if "error" not in result:
            # Display success message with details
            chunks_created = result.get("chunks_created", 0)
            execution_time = result.get("execution_time", 0)
            
            st.success(f"✅ Text content processed successfully! {chunks_created} chunks created in {execution_time:.2f}s")
            
            # Show detailed results
            if result.get("details"):
                with st.expander("📊 Processing Details", expanded=False):
                    details = result["details"]
                    st.write(f"**Content Length:** {details.get('content_length', 0)} characters")
                    st.write(f"**Chunks Created:** {details.get('chunks_created', 0)}")
                    st.write(f"**Vector Storage:** {'✅ Stored' if details.get('vector_stored', False) else '⏭️ Skipped'}")
                    st.write(f"**Knowledge Type:** {details.get('knowledge_type', 'unknown')}")
            
            return True
        else:
            st.error(f"Error processing text content: {result['error']}")
            return False
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def query_knowledge(query: str, thread_id: str = None):
    """Query the knowledge base"""
    if not query.strip():
        st.error("Please enter a query")
        return None
    
    try:
        data = {
            "query": query.strip(),
            "thread_id": thread_id if thread_id else None
        }
        
        result = api_request(API_ENDPOINTS["query"], "POST", data)
        
        if "error" not in result:
            return {
                "answer": result.get("result", {}).get("generated_answer", "No answer generated"),
                "execution_time": result.get("execution_time", 0),
                "thread_id": result.get("thread_id", "unknown"),
                "retrieved_docs": result.get("result", {}).get("retrieved_docs", [])
            }
        else:
            st.error(f"Error processing query: {result['error']}")
            return None
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def is_idk_response(response_text: str) -> bool:
    """Check if the AI response indicates it doesn't know something"""
    idk_phrases = [
        "i don't know",
        "i don't know based on available knowledge",
        "based on available knowledge",
        "insufficient information",
        "no information found",
        "not mentioned in",
        "not available in",
        "cannot find",
        "unable to find"
    ]
    
    response_lower = response_text.lower()
    return any(phrase in response_lower for phrase in idk_phrases)

def submit_feedback(thread_id: str, feedback: str, edits: str = None, comment: str = None, knowledge_type: str = None):
    """Submit feedback for an AI-generated answer"""
    try:
        data = {
            "thread_id": thread_id,
            "feedback": feedback,
            "edits": edits,
            "comment": comment,
            "knowledge_type": knowledge_type
        }
        
        result = api_request(API_ENDPOINTS["feedback"], "POST", data)
        
        if "error" not in result:
            return {
                "success": result.get("success", False),
                "message": result.get("message", "Feedback submitted"),
                "action_taken": result.get("action_taken", "unknown")
            }
        else:
            st.error(f"Error submitting feedback: {result['error']}")
            return None
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def submit_knowledge(thread_id: str, user_knowledge: str, original_query: str):
    """Submit user-provided knowledge when AI doesn't know something"""
    try:
        # Use the ingest endpoint to store the knowledge
        data = {
            "document": user_knowledge,
            "source": f"User Knowledge for: {original_query}",
            "categories": ["user_provided"],
            "author": "user",
            "knowledge_type": "verified"
        }
        
        result = api_request(API_ENDPOINTS["ingest"], "POST", data)
        
        if "error" not in result:
            return {
                "success": result.get("success", False),
                "message": "Knowledge stored successfully",
                "action_taken": "User knowledge added to database"
            }
        else:
            st.error(f"Error storing knowledge: {result['error']}")
            return None
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def get_feedback_status(thread_id: str):
    """Get feedback status for a thread"""
    try:
        endpoint = f"{API_ENDPOINTS['feedback_status']}/{thread_id}"
        result = api_request(endpoint, "GET")
        
        if "error" not in result:
            return {
                "status": result.get("status", "unknown"),
                "has_pending_feedback": result.get("has_pending_feedback", False),
                "current_answer": result.get("current_answer"),
                "feedback_history": result.get("feedback_history", [])
            }
        else:
            return None
            
    except Exception as e:
        return None

# Main App
def main():
    # Header
    col1, col2 = st.columns([0.15, 0.85])
    
    with col1:
        try:
            st.image("second-brain.jpeg", width=60)
        except Exception as e:
            # Fallback to emoji if image fails
            st.markdown("""
            <div style="font-size: 60px; text-align: center; line-height: 60px;">
                🧠
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="d-flex align-items-center">
            <h1 class="mb-0">Smart Second Brain</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Health Status (Compact)
    check_system_health()
    health = st.session_state.system_health
    
    health_color = "status-healthy" if health['health'] == 'healthy' else "status-error" if health['health'] == 'error' else "status-warning"
    
    st.markdown(f"""
    <div class="health-status">
        <div class="d-flex justify-content-between align-items-center">
            <div class="d-flex align-items-center">
                <span class="status-indicator {health_color}"></span>
                <strong>System: {health['health'].title()}</strong>
                <span class="status-indicator {'status-healthy' if health['graph_initialized'] else 'status-error'} ms-3"></span>
                <span class="me-3">Graph</span>
                <span class="status-indicator {'status-healthy' if health['vectorstore_ready'] else 'status-error'}"></span>
                <span class="me-3">Vector</span>
                <span class="status-indicator {'status-healthy' if health['llm_ready'] else 'status-error'}"></span>
                <span>AI</span>
            </div>
            <small class="text-muted">{health.get('last_check', 'Never')}</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📄 PDF Ingestion", "📝 Text Ingestion", "💬 Chat with Knowledge Base"])
    
    with tab1:
        st.markdown("### Upload and Process Documents")
        
        st.markdown("#### 📁 Upload PDF Files")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload multiple PDF files to process"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            # Display uploaded files
            st.markdown("**Uploaded Files:**")
            for i, file in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f"""
                    <div class="file-item">
                        <div>
                            <strong>{file.name}</strong><br>
                            <small class="text-muted">{file.size} bytes</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # PDF metadata form (always visible)
        st.markdown("**PDF Batch Metadata:**")
        
        # Create columns for the form fields
        col1, col2 = st.columns(2)
        
        with col1:
            pdf_source = st.text_input(
                "Source *", 
                placeholder="e.g., research papers, company docs, training materials", 
                key="pdf_source",
                help="Required: Source identifier for all PDFs in this batch"
            )
            pdf_categories = st.text_input(
                "Categories (comma-separated)", 
                placeholder="e.g., ai, research, tutorial, legal", 
                key="pdf_categories",
                help="Optional: Categories to tag all PDFs in this batch"
            )
        
        with col2:
            pdf_author = st.text_input(
                "Author/Organization", 
                placeholder="Document author or organization", 
                key="pdf_author",
                help="Optional: Author or organization for all PDFs"
            )
            pdf_metadata = st.text_input(
                "Additional Metadata", 
                placeholder="e.g., project: AI research, version: 1.0", 
                key="pdf_metadata",
                help="Optional: Additional metadata as JSON string"
            )
        
        # Batch processing info
        if uploaded_files:
            st.markdown(f"""
            <div class="alert alert-info" role="alert">
                <strong>📋 Batch Processing:</strong> {len(uploaded_files)} PDF files will be processed together with the same metadata.
                <br><small>Each PDF will be chunked and embedded into the knowledge base.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert alert-warning" role="alert">
                <strong>📁 No Files Selected:</strong> Please upload PDF files above to enable batch processing.
            </div>
            """, unsafe_allow_html=True)
        
        # Process button (not in a form)
        if st.button(
            "🚀 Process PDF Batch" if uploaded_files else "📁 Upload Files First", 
            type="primary",
            disabled=not uploaded_files,
            key="pdf_process_button"
        ):
            if uploaded_files:
                if not pdf_source.strip():
                    st.error("❌ Please provide a source for the PDF files")
                else:
                    with st.spinner("🔄 Processing PDF files..."):
                        success = ingest_multiple_pdfs(
                            uploaded_files, 
                            pdf_source, 
                            pdf_categories, 
                            pdf_author, 
                            pdf_metadata
                        )
                        
                        if success:
                            # Clear the uploaded files after successful processing
                            st.session_state.uploaded_files = []
                            st.rerun()
    
    with tab2:
        st.markdown("### Text Content Ingestion")
        
        st.markdown("#### 📝 Enter Text Content")
        
        # Text input area
        text_content = st.text_area(
            "Enter text content to ingest:",
            placeholder="Paste or type text content here...",
            height=200,
            help="Enter any text content that you want to add to the knowledge base"
        )
        
        # Text metadata form
        st.markdown("**Text Content Metadata:**")
        
        # Create columns for the form fields
        col1, col2 = st.columns(2)
        
        with col1:
            text_source = st.text_input(
                "Source *", 
                placeholder="e.g., manual, notes, article, documentation", 
                key="text_source",
                help="Required: Source identifier for this text content"
            )
            text_categories = st.text_input(
                "Categories (comma-separated)", 
                placeholder="e.g., ai, research, tutorial, legal", 
                key="text_categories",
                help="Optional: Categories to tag this text content"
            )
        
        with col2:
            text_author = st.text_input(
                "Author", 
                placeholder="e.g., John Doe, OpenAI, Company Name", 
                key="text_author",
                help="Optional: Author or creator of the content"
            )
            text_knowledge_type = st.selectbox(
                "Knowledge Type",
                ["reusable", "verified", "temporary"],
                key="text_knowledge_type",
                help="Type of knowledge: reusable (long-term), verified (validated), temporary (short-term)"
            )
        
        # Process button
        if st.button("🚀 Process Text Content", key="process_text"):
            if not text_content.strip():
                st.error("❌ Please enter some text content to process.")
            elif not text_source.strip():
                st.error("❌ Please provide a source for the text content.")
            else:
                # Show processing status
                with st.spinner("🔄 Processing text content..."):
                    success = process_text_ingestion(
                        text_content,
                        text_source,
                        text_categories,
                        text_author,
                        text_knowledge_type
                    )
                    
                    if success:
                        st.success("✅ Text content processed successfully!")
                        st.rerun()
    
    with tab3:
        st.markdown("### Chat with Your Knowledge Base")
        
        # Thread ID display
        if st.session_state.current_thread_id:
            st.info(f"🔄 **Current Conversation Thread:** `{st.session_state.current_thread_id}`")
        else:
            st.info("💬 **New Conversation** - A thread ID will be created when you send your first message")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history):
                if chat["type"] == "user":
                    st.markdown(f"""
                    <div class="d-flex justify-content-end mb-2">
                        <div class="chat-message user-message">
                            {chat["content"]}
                            <br><small style="opacity: 0.8;">{chat["timestamp"]}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="d-flex justify-content-start mb-2">
                        <div class="chat-message assistant-message">
                            {chat["content"]}
                            <br><small class="text-muted">
                                {chat["timestamp"]}
                                {f" • {chat.get('execution_time', 0):.2f}s" if chat.get('execution_time') else ""}
                                {f" • Thread: {chat.get('thread_id', 'unknown')}" if chat.get('thread_id') else ""}
                            </small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add feedback interface for assistant messages
                    if not chat.get("error", False):
                        with st.container():
                            st.markdown("---")
                            
                            # Check if this is an "I don't know" response
                            is_idk = is_idk_response(chat["content"])
                            
                            if is_idk:
                                # Special interface for "I don't know" responses
                                st.info("🤔 The AI doesn't know about this topic. Help improve the system by providing knowledge!")
                                
                                # Find the original user query for this response
                                original_query = ""
                                for j in range(i-1, -1, -1):
                                    if st.session_state.chat_history[j]["type"] == "user":
                                        original_query = st.session_state.chat_history[j]["content"]
                                        break
                                
                                # Knowledge input interface
                                if st.session_state.get(f"providing_knowledge_{i}", False):
                                    st.markdown("**Provide knowledge about this topic:**")
                                    user_knowledge = st.text_area(
                                        f"Knowledge about: {original_query}",
                                        placeholder="Enter what you know about this topic...",
                                        key=f"knowledge_input_{i}",
                                        height=120
                                    )
                                    
                                    col_knowledge1, col_knowledge2 = st.columns([0.3, 0.7])
                                    with col_knowledge1:
                                        if st.button("💾 Store Knowledge", key=f"store_knowledge_{i}"):
                                            if user_knowledge.strip():
                                                if submit_knowledge(chat.get("thread_id", ""), user_knowledge.strip(), original_query):
                                                    st.session_state.chat_history[i]["feedback_status"] = "knowledge_provided"
                                                    st.session_state[f"providing_knowledge_{i}"] = False
                                                    st.success("Knowledge stored! The system will remember this for future questions.")
                                                    st.rerun()
                                            else:
                                                st.warning("Please enter some knowledge before storing.")
                                    
                                    with col_knowledge2:
                                        if st.button("❌ Cancel", key=f"cancel_knowledge_{i}"):
                                            st.session_state[f"providing_knowledge_{i}"] = False
                                            st.rerun()
                                else:
                                    col_knowledge_btn, col_status = st.columns([0.3, 0.7])
                                    with col_knowledge_btn:
                                        if st.button("📚 Provide Knowledge", key=f"provide_knowledge_{i}"):
                                            st.session_state[f"providing_knowledge_{i}"] = True
                                            st.rerun()
                                    
                                    with col_status:
                                        feedback_status = chat.get("feedback_status", "pending")
                                        if feedback_status == "knowledge_provided":
                                            st.success("✅ Knowledge provided")
                                        else:
                                            st.info("⏳ Waiting for knowledge input")
                            
                            else:
                                # Standard feedback interface for regular responses
                                col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.2, 0.4])
                                
                                with col1:
                                    if st.button("👍 Approve", key=f"approve_{i}", help="Mark this answer as approved"):
                                        if submit_feedback(chat.get("thread_id", ""), "approved"):
                                            st.session_state.chat_history[i]["feedback_status"] = "approved"
                                            st.success("Answer approved!")
                                            st.rerun()
                                
                                with col2:
                                    if st.button("👎 Reject", key=f"reject_{i}", help="Mark this answer as rejected"):
                                        if submit_feedback(chat.get("thread_id", ""), "rejected"):
                                            st.session_state.chat_history[i]["feedback_status"] = "rejected"
                                            st.success("Answer rejected!")
                                            st.rerun()
                                
                                with col3:
                                    if st.button("✏️ Edit", key=f"edit_{i}", help="Provide edited version"):
                                        # Store the message index for editing
                                        st.session_state[f"editing_message_{i}"] = True
                                        st.rerun()
                                
                                with col4:
                                    # Show current feedback status
                                    feedback_status = chat.get("feedback_status", "pending")
                                    if feedback_status == "approved":
                                        st.success("✅ Approved")
                                    elif feedback_status == "rejected":
                                        st.error("❌ Rejected")
                                    elif feedback_status == "edited":
                                        st.info("✏️ Edited")
                                    else:
                                        st.info("⏳ Pending feedback")
                                
                                # Show edit interface if editing
                                if st.session_state.get(f"editing_message_{i}", False):
                                    st.markdown("**Edit the answer:**")
                                    edited_content = st.text_area(
                                        "Edited answer:",
                                        value=chat["content"],
                                        key=f"edit_content_{i}",
                                        height=100
                                    )
                                    
                                    col_edit1, col_edit2 = st.columns([0.3, 0.7])
                                    with col_edit1:
                                        if st.button("Submit Edit", key=f"submit_edit_{i}"):
                                            if submit_feedback(chat.get("thread_id", ""), "edited", edits=edited_content):
                                                st.session_state.chat_history[i]["content"] = edited_content
                                                st.session_state.chat_history[i]["feedback_status"] = "edited"
                                                st.session_state[f"editing_message_{i}"] = False
                                                st.success("Answer updated!")
                                                st.rerun()
                                    
                                    with col_edit2:
                                        if st.button("Cancel", key=f"cancel_edit_{i}"):
                                            st.session_state[f"editing_message_{i}"] = False
                                            st.rerun()
        
        # New Thread button (outside form)
        col_new_thread, col_spacer = st.columns([1, 4])
        with col_new_thread:
            if st.button("🔄 New Thread", help="Start a new conversation thread"):
                st.session_state.current_thread_id = None
                st.session_state.new_thread_clicked = True
                st.rerun()
        
        # Chat input
        with st.form("chat_form"):
            col_input, col_button = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Type your question...",
                    placeholder="Ask anything about your knowledge base...",
                    label_visibility="collapsed"
                )
            
            with col_button:
                send_button = st.form_submit_button("Send", type="primary")
            
            # Advanced options
            with st.expander("Advanced Options"):
                # Clear thread ID if new thread was clicked
                if st.session_state.get('new_thread_clicked', False):
                    thread_id_value = ""
                    st.session_state.new_thread_clicked = False
                else:
                    thread_id_value = st.session_state.current_thread_id or ""
                
                thread_id = st.text_input(
                    "Thread ID (optional)", 
                    value=thread_id_value,
                    placeholder="For conversation continuity",
                    help="Leave empty to start a new conversation or continue with current thread"
                )
        
        # Handle chat submission
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "type": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Get AI response
            with st.spinner("Thinking..."):
                # Use provided thread_id or current thread_id (but only if not explicitly cleared)
                if thread_id.strip():
                    active_thread_id = thread_id.strip()
                elif st.session_state.current_thread_id:
                    active_thread_id = st.session_state.current_thread_id
                else:
                    active_thread_id = None  # Start new conversation
                result = query_knowledge(user_input, active_thread_id)
                
                if result:
                    # Store the thread ID from the response
                    if result.get("thread_id"):
                        st.session_state.current_thread_id = result["thread_id"]
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "type": "assistant",
                        "content": result["answer"],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "execution_time": result["execution_time"],
                        "retrieved_docs": result["retrieved_docs"],
                        "thread_id": result.get("thread_id", "unknown"),
                        "feedback_status": "pending"  # Mark as pending feedback
                    })
                else:
                    # Add error message
                    st.session_state.chat_history.append({
                        "type": "assistant",
                        "content": "Sorry, I encountered an error processing your request.",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "error": True
                    })
            
            st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.current_thread_id = None
            st.rerun()

if __name__ == "__main__":
    main()