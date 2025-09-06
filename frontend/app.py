#!/usr/bin/env python3
"""
Smart Second Brain - Streamlit Frontend
A modern web interface for AI-powered knowledge management
"""

import asyncio
import json
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional
import io

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

def ingest_document(content: str, source: str = "", categories: str = "", author: str = ""):
    """Ingest a document into the knowledge base"""
    if not content.strip():
        st.error("Please enter document content")
        return False
    
    try:
        data = {
            "document": content.strip(),
            "source": source or "manual_input",
            "categories": [cat.strip() for cat in categories.split(",") if cat.strip()] if categories else [],
            "metadata": {
                "author": author or "User",
                "date": datetime.now().isoformat(),
                "frontend": "streamlit"
            }
        }
        
        result = api_request(API_ENDPOINTS["ingest"], "POST", data)
        
        if "error" not in result:
            st.success("Document ingested successfully!")
            return True
        else:
            st.error(f"Error ingesting document: {result['error']}")
            return False
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

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

# Main App
def main():
    # Header
    col1, col2, col3 = st.columns([0.1, 0.6, 0.3])
    
    with col1:
        st.image("second-brain.jpeg", width=40)
    
    with col2:
        st.markdown("""
        <div class="d-flex align-items-center">
            <h1 class="me-3 mb-0">Smart Second Brain</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="d-flex align-items-center h-100">
            <span class="text-muted">AI-Powered Knowledge Management</span>
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
    tab1, tab2 = st.tabs(["📄 Document Ingestion", "💬 Chat with Knowledge Base"])
    
    with tab1:
        st.markdown("### Upload and Process Documents")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
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
        
        with col2:
            st.markdown("#### ✏️ Manual Document Input")
            
            # Manual input form
            with st.form("manual_ingest"):
                content = st.text_area(
                    "Document Content",
                    placeholder="Paste or type your document content here...",
                    height=200
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    source = st.text_input("Source", placeholder="e.g., webpage, document, manual")
                with col_b:
                    categories = st.text_input("Categories (comma-separated)", placeholder="e.g., ai, research, tutorial")
                
                author = st.text_input("Author", placeholder="Your name or organization")
                
                submitted = st.form_submit_button("Ingest Document", type="primary")
                
                if submitted:
                    with st.spinner("Processing document..."):
                        success = ingest_document(content, source, categories, author)
                        if success:
                            st.rerun()
    
    with tab2:
        st.markdown("### Chat with Your Knowledge Base")
        
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
                                {f" • {len(chat.get('retrieved_docs', []))} docs" if chat.get('retrieved_docs') else ""}
                            </small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show retrieved documents
                    if chat.get("retrieved_docs") and not chat.get("error", False):
                        for j, doc in enumerate(chat["retrieved_docs"][:2]):
                            st.markdown(f"""
                            <div class="ms-4 mb-2 p-2 bg-light border-start border-primary border-3 rounded">
                                <small class="text-muted fw-bold">Source {j+1}: {doc.get('metadata', {}).get('source', 'unknown')}</small>
                                <p class="mb-0 small">{doc.get('content', '')[:150]}{'...' if len(doc.get('content', '')) > 150 else ''}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
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
                thread_id = st.text_input("Thread ID (optional)", placeholder="For conversation continuity")
        
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
                result = query_knowledge(user_input, thread_id)
                
                if result:
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "type": "assistant",
                        "content": result["answer"],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "execution_time": result["execution_time"],
                        "retrieved_docs": result["retrieved_docs"]
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
            st.rerun()

if __name__ == "__main__":
    main()