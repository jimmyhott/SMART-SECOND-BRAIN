#!/usr/bin/env python3
"""
Simple test script to verify OpenAI API connection.
This will help diagnose API connectivity issues.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("openai_test")

def test_openai_direct():
    """Test direct OpenAI API connection."""
    logger.info("🔍 Testing OpenAI API Connection")
    
    try:
        import openai
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment")
            return False
        
        logger.info(f"🔑 API Key found: {api_key[:10]}...{api_key[-4:]}")
        
        # Get model name from environment
        model_name = os.getenv("LLM_MODEL", "gpt-4")
        logger.info(f"🤖 Using model: {model_name}")
        
        # Configure OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Test simple completion
        logger.info("📡 Sending test request to OpenAI...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from Smart Second Brain!' and nothing else."}
            ],
            max_tokens=50,
            temperature=0
        )
        
        logger.info("✅ OpenAI API test successful!")
        logger.info(f"🤖 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenAI API test failed: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        return False

def test_azure_openai():
    """Test Azure OpenAI API connection."""
    logger.info("🔍 Testing Azure OpenAI API Connection")
    
    try:
        import openai
        
        # Get Azure configuration
        api_key = os.getenv("OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
        
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment")
            return False
            
        if not azure_endpoint or azure_endpoint == "https://your-resource-name.openai.azure.com/":
            logger.warning("⚠️ AZURE_OPENAI_ENDPOINT_URL not configured or is placeholder")
            return False
        
        logger.info(f"🔑 API Key found: {api_key[:10]}...{api_key[-4:]}")
        logger.info(f"🌐 Azure Endpoint: {azure_endpoint}")
        
        # Get API version from environment
        api_version = os.getenv("API_VERSION", "2024-02-15-preview")
        
        # Configure Azure OpenAI client
        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        
        # Get model name from environment
        model_name = os.getenv("LLM_MODEL", "gpt-4")
        logger.info(f"🤖 Using model: {model_name}")
        
        # Test simple completion
        logger.info("📡 Sending test request to Azure OpenAI...")
        response = client.chat.completions.create(
            model=model_name,  # Use the model from environment
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from Smart Second Brain!' and nothing else."}
            ],
            max_tokens=50,
            temperature=0
        )
        
        logger.info("✅ Azure OpenAI API test successful!")
        logger.info(f"🤖 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Azure OpenAI API test failed: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        return False

def test_langchain_openai():
    """Test LangChain OpenAI integration."""
    logger.info("🔍 Testing LangChain OpenAI Integration")
    
    try:
        from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
        
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment")
            return False
        
        logger.info(f"🔑 API Key found: {api_key[:10]}...{api_key[-4:]}")
        
        # Get model name from environment
        model_name = os.getenv("LLM_MODEL", "gpt-4")
        
        # Configure LangChain LLM
        if azure_endpoint and azure_endpoint != "https://your-resource-name.openai.azure.com/":
            logger.info(f"🌐 Using Azure OpenAI endpoint: {azure_endpoint}")
            logger.info(f"🤖 Using model: {model_name}")
            
            # Get API version from environment
            api_version = os.getenv("API_VERSION", "2024-02-15-preview")
            
            # Use AzureChatOpenAI for Azure endpoints
            llm = AzureChatOpenAI(
                azure_deployment=model_name,  # This is the deployment name
                openai_api_version=api_version,
                azure_endpoint=azure_endpoint,
                openai_api_key=api_key,
                temperature=0
            )
        else:
            logger.info("🌐 Using OpenAI API directly")
            logger.info(f"🤖 Using model: {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                openai_api_key=api_key
            )
        
        # Test LangChain integration
        logger.info("📡 Sending test request via LangChain...")
        response = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from Smart Second Brain!' and nothing else."}
        ])
        
        logger.info("✅ LangChain OpenAI integration test successful!")
        logger.info(f"🤖 Response: {response.content}")
        return True
        
    except Exception as e:
        logger.error(f"❌ LangChain OpenAI integration test failed: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        return False

def test_langchain_embeddings():
    """Test LangChain embeddings integration."""
    logger.info("🔍 Testing LangChain Embeddings Integration")
    
    try:
        from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
        
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment")
            return False
        
        logger.info(f"🔑 API Key found: {api_key[:10]}...{api_key[-4:]}")
        
        # Get embedding model from environment
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Configure LangChain embeddings
        if azure_endpoint and azure_endpoint != "https://your-resource-name.openai.azure.com/":
            logger.info(f"🌐 Using Azure OpenAI embeddings: {embedding_model}")
            
            # Get API version from environment
            api_version = os.getenv("API_VERSION", "2024-02-15-preview")
            
            # Use AzureOpenAIEmbeddings for Azure endpoints
            deployment = "text-embedding-3-small"
            logger.info(f"🚀 Using deployment: {deployment}")
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=deployment,
                openai_api_version=api_version,
                azure_endpoint=azure_endpoint,
                openai_api_key=api_key
            )
        else:
            logger.info("🌐 Using OpenAI embeddings directly")
            logger.info(f"🔤 Using model: {embedding_model}")
            embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=api_key
            )
        
        # Test embeddings
        logger.info("📡 Sending test embedding request via LangChain...")
        test_texts = ["Hello from Smart Second Brain!", "This is a test document."]
        embeddings_result = embeddings.embed_documents(test_texts)
        
        logger.info("✅ LangChain embeddings integration test successful!")
        logger.info(f"🔤 Generated {len(embeddings_result)} embeddings")
        logger.info(f"🔤 Embedding dimensions: {len(embeddings_result[0])}")
        return True
        
    except Exception as e:
        logger.error(f"❌ LangChain embeddings integration test failed: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        return False

def main():
    """Run all API connection tests."""
    logger.info("🚀 Starting OpenAI API Connection Tests")
    logger.info("=" * 50)
    
    # Test 1: Direct OpenAI API
    logger.info("\n1️⃣ Testing Direct OpenAI API")
    openai_success = test_openai_direct()
    
    # Test 2: Azure OpenAI API
    logger.info("\n2️⃣ Testing Azure OpenAI API")
    azure_success = test_azure_openai()
    
    # Test 3: LangChain Integration
    logger.info("\n3️⃣ Testing LangChain Integration")
    langchain_success = test_langchain_openai()
    
    # Test 4: LangChain Embeddings
    logger.info("\n4️⃣ Testing LangChain Embeddings")
    embeddings_success = test_langchain_embeddings()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    logger.info(f"   Direct OpenAI API: {'✅ PASS' if openai_success else '❌ FAIL'}")
    logger.info(f"   Azure OpenAI API: {'✅ PASS' if azure_success else '❌ FAIL'}")
    logger.info(f"   LangChain Integration: {'✅ PASS' if langchain_success else '❌ FAIL'}")
    logger.info(f"   LangChain Embeddings: {'✅ PASS' if embeddings_success else '❌ FAIL'}")
    
    if not any([openai_success, azure_success, langchain_success, embeddings_success]):
        logger.error("\n❌ All API tests failed!")
        logger.error("🔧 Troubleshooting suggestions:")
        logger.error("   1. Check your OPENAI_API_KEY is valid")
        logger.error("   2. Verify your AZURE_OPENAI_ENDPOINT_URL (if using Azure)")
        logger.error("   3. Check your internet connection")
        logger.error("   4. Verify OpenAI service is available in your region")
        logger.error("   5. Try using a VPN if you're in a restricted region")
        return False
    else:
        logger.info("\n✅ At least one API connection is working!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
