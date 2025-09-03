#!/usr/bin/env python3
"""
Smart Second Brain - NiceGUI Frontend
A modern web interface for AI-powered knowledge management
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from nicegui import app, ui, run

# Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/health",
    "ingest": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/ingest",
    "query": f"{API_BASE_URL}/smart-second-brain/api/v1/graph/query",
}

# Global state
system_status = {
    "health": "unknown",
    "graph_initialized": False,
    "vectorstore_ready": False,
    "embedding_model_ready": False,
    "llm_ready": False,
    "last_check": None
}

recent_ingestions = []
recent_queries = []

# Global UI elements (will be set in main_page)
ingestions_display = None
queries_display = None
query_results = None
document_content = None
source_input = None
categories_input = None
author_input = None
query_input = None
thread_id_input = None
ingest_button = None
query_button = None

# API Client
async def api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
    """Make API request to backend"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                response = await client.get(endpoint)
            elif method == "POST":
                response = await client.post(endpoint, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}

# System Health Functions
async def check_system_health():
    """Check system health status"""
    global system_status
    
    try:
        result = await api_request(API_ENDPOINTS["health"])
        
        if "error" not in result:
            system_status.update({
                "health": result.get("status", "unknown"),
                "graph_initialized": result.get("graph_initialized", False),
                "vectorstore_ready": result.get("vectorstore_ready", False),
                "embedding_model_ready": result.get("embedding_model_ready", False),
                "llm_ready": result.get("llm_ready", False),
                "last_check": datetime.now().strftime("%H:%M:%S")
            })
        else:
            system_status["health"] = "error"
            
    except Exception as e:
        system_status["health"] = "error"
        system_status["last_check"] = datetime.now().strftime("%H:%M:%S")

async def auto_refresh_health():
    """Auto-refresh health status every 10 seconds"""
    while True:
        await check_system_health()
        await asyncio.sleep(10)

# Document Ingestion Functions
async def ingest_document():
    """Ingest a document into the knowledge base"""
    global recent_ingestions
    
    if not document_content.value.strip():
        ui.notify("Please enter document content", type="warning")
        return
    
    # Show loading state
    ingest_button.disable()
    ingest_button.text = "Processing..."
    
    try:
        data = {
            "document": document_content.value.strip(),
            "source": source_input.value or "manual_input",
            "categories": [cat.strip() for cat in categories_input.value.split(",") if cat.strip()] if categories_input.value else [],
            "metadata": {
                "author": author_input.value or "User",
                "date": datetime.now().isoformat(),
                "frontend": "nicegui"
            }
        }
        
        result = await api_request(API_ENDPOINTS["ingest"], "POST", data)
        
        if "error" not in result:
            # Add to recent ingestions
            recent_ingestions.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "status": result.get("result", {}).get("status", "completed"),
                "execution_time": result.get("execution_time", 0),
                "thread_id": result.get("thread_id", "unknown"),
                "content": document_content.value[:100] + "..." if len(document_content.value) > 100 else document_content.value
            })
            
            # Keep only last 10
            recent_ingestions = recent_ingestions[:10]
            
            # Clear form
            document_content.value = ""
            source_input.value = ""
            categories_input.value = ""
            author_input.value = ""
            
            ui.notify("Document ingested successfully!", type="positive")
            update_ingestions_display()
        else:
            ui.notify(f"Error ingesting document: {result['error']}", type="negative")
            
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")
    
    finally:
        # Reset button
        ingest_button.enable()
        ingest_button.text = "Ingest Document"

# Knowledge Query Functions
async def query_knowledge():
    """Query the knowledge base"""
    global recent_queries
    
    if not query_input.value.strip():
        ui.notify("Please enter a query", type="warning")
        return
    
    # Show loading state
    query_button.disable()
    query_button.text = "Processing..."
    
    try:
        data = {
            "query": query_input.value.strip(),
            "thread_id": thread_id_input.value if thread_id_input.value else None
        }
        
        result = await api_request(API_ENDPOINTS["query"], "POST", data)
        
        if "error" not in result:
            # Add to recent queries
            recent_queries.insert(0, {
                "timestamp": datetime.now().isoformat(),
                "query": query_input.value,
                "answer": result.get("result", {}).get("generated_answer", "No answer generated"),
                "execution_time": result.get("execution_time", 0),
                "thread_id": result.get("thread_id", "unknown"),
                "retrieved_docs": len(result.get("result", {}).get("retrieved_docs", []))
            })
            
            # Keep only last 10
            recent_queries = recent_queries[:10]
            
            # Display results
            query_results.clear()
            with query_results:
                ui.html(f"""
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #495057;">AI Response:</h4>
                        <p style="margin: 0; line-height: 1.6;">{result.get("result", {}).get("generated_answer", "No answer generated")}</p>
                    </div>
                """)
                
                if result.get("result", {}).get("retrieved_docs"):
                    ui.html(f"""
                        <div style="padding: 1rem; background: #e9ecef; border-radius: 8px; margin: 1rem 0;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #495057;">Retrieved Documents ({len(result.get("result", {}).get("retrieved_docs", []))}):</h4>
                        </div>
                    """)
                    
                    for i, doc in enumerate(result.get("result", {}).get("retrieved_docs", [])[:3]):
                        ui.html(f"""
                            <div style="padding: 0.75rem; background: white; border: 1px solid #dee2e6; border-radius: 6px; margin: 0.5rem 0;">
                                <strong>Document {i+1}:</strong> {doc.get("content", "")[:200]}...
                                <br><small style="color: #6c757d;">Source: {doc.get("metadata", {}).get("source", "unknown")}</small>
                            </div>
                        """)
            
            # Clear query input
            query_input.value = ""
            
            ui.notify("Query processed successfully!", type="positive")
            update_queries_display()
        else:
            ui.notify(f"Error processing query: {result['error']}", type="negative")
            
    except Exception as e:
        ui.notify(f"Error: {str(e)}", type="negative")
    
    finally:
        # Reset button
        query_button.enable()
        query_button.text = "Query Knowledge"

# UI Update Functions
def update_ingestions_display():
    """Update the recent ingestions display"""
    if ingestions_display:
        ingestions_display.clear()
        
        if not recent_ingestions:
            with ingestions_display:
                ui.html('<p style="text-align: center; color: #6c757d; padding: 2rem;">No documents ingested yet</p>')
            return
        
        with ingestions_display:
            for ingestion in recent_ingestions:
                ui.html(f"""
                    <div style="padding: 0.75rem; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div style="flex: 1;">
                                <p style="margin: 0; font-weight: 500; color: #495057;">{ingestion['status']}</p>
                                <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; color: #6c757d;">{ingestion['content']}</p>
                                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; color: #6c757d;">{ingestion['timestamp']}</p>
                            </div>
                            <span style="font-size: 0.75rem; color: #6c757d;">{ingestion['execution_time']:.2f}s</span>
                        </div>
                    </div>
                """)

def update_queries_display():
    """Update the recent queries display"""
    if queries_display:
        queries_display.clear()
        
        if not recent_queries:
            with queries_display:
                ui.html('<p style="text-align: center; color: #6c757d; padding: 2rem;">No queries made yet</p>')
            return
        
        with queries_display:
            for query in recent_queries:
                ui.html(f"""
                    <div style="padding: 0.75rem; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div style="flex: 1;">
                                <p style="margin: 0; font-weight: 500; color: #495057;">{query['query'][:50]}{'...' if len(query['query']) > 50 else ''}</p>
                                <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; color: #6c757d;">{query['answer'][:100]}{'...' if len(query['answer']) > 100 else ''}</p>
                                <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; color: #6c757d;">{query['timestamp']} • {query['retrieved_docs']} docs</p>
                            </div>
                            <span style="font-size: 0.75rem; color: #6c757d;">{query['execution_time']:.2f}s</span>
                        </div>
                    </div>
                """)

def get_status_color(status: str) -> str:
    """Get color for status indicators"""
    if status == "healthy":
        return "#28a745"
    elif status == "error":
        return "#dc3545"
    else:
        return "#ffc107"

# Main UI Layout
@ui.page('/')
def main_page():
    """Main application page"""
    global ingestions_display, queries_display, query_results
    global document_content, source_input, categories_input, author_input
    global query_input, thread_id_input, ingest_button, query_button
    
    # Header
    with ui.header().classes('bg-blue-600 text-white'):
        ui.label('🧠 Smart Second Brain').classes('text-h4 q-mr-md')
        ui.label('AI-Powered Knowledge Management').classes('text-subtitle2')
        
        with ui.row().classes('q-ml-auto'):
            ui.button('🔄 Refresh Health', on_click=check_system_health).classes('q-mr-sm')
            ui.button('📚 API Docs', on_click=lambda: ui.open(API_BASE_URL + '/docs')).classes('q-mr-sm')
    
    # Main content
    with ui.column().classes('full-width q-pa-md'):
        
        # System Health Section
        with ui.card().classes('full-width q-mb-md'):
            ui.label('🏥 System Health').classes('text-h6 q-mb-md')
            
            with ui.row().classes('full-width'):
                with ui.column().classes('col-3'):
                    ui.html(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 60px; height: 60px; border-radius: 50%; background: {get_status_color(system_status['health'])}; margin: 0 auto 0.5rem auto; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                                {'✅' if system_status['health'] == 'healthy' else '❌' if system_status['health'] == 'error' else '⚠️'}
                            </div>
                            <p style="margin: 0; font-weight: 500;">Overall</p>
                            <p style="margin: 0; font-size: 0.875rem; color: #6c757d;">{system_status['health'].title()}</p>
                        </div>
                    """)
                
                with ui.column().classes('col-3'):
                    ui.html(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 60px; height: 60px; border-radius: 50%; background: {'#28a745' if system_status['graph_initialized'] else '#dc3545'}; margin: 0 auto 0.5rem auto; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                                {'✅' if system_status['graph_initialized'] else '❌'}
                            </div>
                            <p style="margin: 0; font-weight: 500;">Graph</p>
                            <p style="margin: 0; font-size: 0.875rem; color: #6c757d;">{'Ready' if system_status['graph_initialized'] else 'Not Ready'}</p>
                        </div>
                    """)
                
                with ui.column().classes('col-3'):
                    ui.html(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 60px; height: 60px; border-radius: 50%; background: {'#28a745' if system_status['vectorstore_ready'] else '#dc3545'}; margin: 0 auto 0.5rem auto; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                                {'✅' if system_status['vectorstore_ready'] else '❌'}
                            </div>
                            <p style="margin: 0; font-weight: 500;">Vector Store</p>
                            <p style="margin: 0; font-size: 0.875rem; color: #6c757d;">{'Ready' if system_status['vectorstore_ready'] else 'Not Ready'}</p>
                        </div>
                    """)
                
                with ui.column().classes('col-3'):
                    ui.html(f"""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 60px; height: 60px; border-radius: 50%; background: {'#28a745' if system_status['llm_ready'] else '#dc3545'}; margin: 0 auto 0.5rem auto; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                                {'✅' if system_status['llm_ready'] else '❌'}
                            </div>
                            <p style="margin: 0; font-weight: 500;">AI Model</p>
                            <p style="margin: 0; font-size: 0.875rem; color: #6c757d;">{'Ready' if system_status['llm_ready'] else 'Not Ready'}</p>
                        </div>
                    """)
            
            if system_status['last_check']:
                ui.html(f'<p style="text-align: center; margin: 1rem 0 0 0; font-size: 0.875rem; color: #6c757d;">Last checked: {system_status["last_check"]}</p>')
        
        # Main Features Grid
        with ui.row().classes('full-width q-col-gutter-md'):
            
            # Document Ingestion
            with ui.column().classes('col-12 col-md-6'):
                with ui.card().classes('full-width'):
                    ui.label('📄 Document Ingestion').classes('text-h6 q-mb-md')
                    
                    # Form inputs
                    document_content = ui.textarea('Document Content', placeholder='Paste or type your document content here...').classes('full-width q-mb-md')
                    document_content.style('min-height: 120px')
                    
                    with ui.row().classes('full-width q-col-gutter-sm'):
                        source_input = ui.input('Source', placeholder='e.g., webpage, document, manual').classes('col-6')
                        categories_input = ui.input('Categories (comma-separated)', placeholder='e.g., ai, research, tutorial').classes('col-6')
                    
                    author_input = ui.input('Author', placeholder='Your name or organization').classes('full-width q-mb-md')
                    
                    ingest_button = ui.button('Ingest Document', on_click=ingest_document).classes('full-width')
                    
                    # Recent ingestions
                    ui.label('Recent Ingestions').classes('text-subtitle1 q-mt-lg q-mb-md')
                    ingestions_display = ui.column().classes('full-width')
                    update_ingestions_display()
            
            # Knowledge Query
            with ui.column().classes('col-12 col-md-6'):
                with ui.card().classes('full-width'):
                    ui.label('🔍 Knowledge Query').classes('text-h6 q-mb-md')
                    
                    # Query inputs
                    query_input = ui.textarea('Your Question', placeholder='Ask anything about your knowledge base...').classes('full-width q-mb-md')
                    query_input.style('min-height: 80px')
                    
                    thread_id_input = ui.input('Thread ID (optional)', placeholder='For conversation continuity').classes('full-width q-mb-md')
                    
                    query_button = ui.button('Query Knowledge', on_click=query_knowledge).classes('full-width')
                    
                    # Query results
                    ui.label('Query Results').classes('text-subtitle1 q-mt-lg q-mb-md')
                    query_results = ui.column().classes('full-width')
                    
                    # Recent queries
                    ui.label('Recent Queries').classes('text-subtitle1 q-mt-lg q-mb-md')
                    queries_display = ui.column().classes('full-width')
                    update_queries_display()
        
        # Quick Start Guide
        with ui.card().classes('full-width q-mt-md'):
            ui.label('🚀 Quick Start Guide').classes('text-h6 q-mb-md')
            
            with ui.row().classes('full-width q-col-gutter-md'):
                with ui.column().classes('col-12 col-md-4'):
                    ui.html("""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 50px; height: 50px; background: #007bff; border-radius: 50%; margin: 0 auto 0.5rem auto; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">1</div>
                            <h5 style="margin: 0 0 0.5rem 0;">Check System Health</h5>
                            <p style="margin: 0; font-size: 0.875rem; color: #6c757d;">Ensure your API is running and all services are healthy</p>
                        </div>
                    """)
                
                with ui.column().classes('col-12 col-md-4'):
                    ui.html("""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 50px; height: 50px; background: #28a745; border-radius: 50%; margin: 0 auto 0.5rem auto; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">2</div>
                            <h5 style="margin: 0 0 0.5rem 0;">Ingest Documents</h5>
                            <p style="margin: 0; font-size: 0.875rem; color: #6c757d;">Upload or paste content to build your knowledge base</p>
                        </div>
                    """)
                
                with ui.column().classes('col-12 col-md-4'):
                    ui.html("""
                        <div style="text-align: center; padding: 1rem;">
                            <div style="width: 50px; height: 50px; background: #ffc107; border-radius: 50%; margin: 0 auto 0.5rem auto; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">3</div>
                            <h5 style="margin: 0 0 0.5rem 0;">Ask Questions</h5>
                            <p style="margin: 0; font-size: 0.875rem; color: #6c757d;">Query your knowledge base with natural language</p>
                        </div>
                    """)

# Initialize the application
async def init_app():
    """Initialize the application"""
    # Check initial health
    await check_system_health()
    
    # Start auto-refresh
    asyncio.create_task(auto_refresh_health())

# Run the application
if __name__ in {"__main__", "__mp_main__"}:
    app.on_startup(init_app)
    ui.run(
        title="Smart Second Brain",
        port=5173,
        show=True,
        reload=False
    )
