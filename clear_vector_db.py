#!/usr/bin/env python3
"""
ChromaDB Vector Database Cleanup Script

This script provides a standalone way to clear all data from the ChromaDB vector database.
Useful for development, testing, and maintenance purposes.

⚠️  WARNING: This operation is irreversible and will delete ALL vector data.

Usage:
    python clear_vector_db.py [--confirm] [--backup]

Options:
    --confirm    Skip confirmation prompt (useful for automation)
    --backup     Create a backup before clearing
    --help       Show this help message

Examples:
    python clear_vector_db.py                    # Interactive mode with confirmation
    python clear_vector_db.py --confirm          # Non-interactive mode
    python clear_vector_db.py --backup --confirm # Create backup then clear
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import chromadb
    from langchain_chroma import Chroma
    from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install chromadb langchain-chroma langchain-openai")
    sys.exit(1)


def load_environment():
    """Load environment variables from .env file if it exists."""
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"✅ Loaded environment from {env_file}")
    else:
        print("⚠️  No .env file found, using system environment variables")


def create_backup(backup_dir: Path) -> bool:
    """Create a backup of the ChromaDB directory."""
    chroma_dir = project_root / "chroma_db"
    
    if not chroma_dir.exists():
        print("ℹ️  No ChromaDB directory found, nothing to backup")
        return True
    
    try:
        print(f"📦 Creating backup to {backup_dir}...")
        shutil.copytree(chroma_dir, backup_dir)
        print(f"✅ Backup created successfully: {backup_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False


def get_embedding_model():
    """Initialize the embedding model based on environment configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
    api_version = os.getenv("API_VERSION", "2024-12-01-preview")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return None
    
    if azure_endpoint and azure_endpoint != "https://your-resource-name.openai.azure.com/":
        # Use Azure OpenAI
        deployment = "text-embedding-3-small"
        print(f"🔗 Using Azure OpenAI embeddings: {deployment}")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=deployment,
            openai_api_version=api_version,
            azure_endpoint=azure_endpoint,
            openai_api_key=api_key
        )
    else:
        # Use OpenAI directly
        print("🔗 Using OpenAI embeddings: text-embedding-3-small")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
    
    return embeddings


def clear_chromadb(collection_name: str = "smart_second_brain", 
                   persist_directory: str = "./chroma_db") -> bool:
    """Clear all data from ChromaDB."""
    try:
        print(f"🗑️  Starting ChromaDB cleanup...")
        print(f"   Collection: {collection_name}")
        print(f"   Directory: {persist_directory}")
        
        # Initialize embedding model
        embedding_model = get_embedding_model()
        if not embedding_model:
            return False
        
        # Initialize ChromaDB client
        chroma_dir = Path(persist_directory)
        if not chroma_dir.exists():
            print("ℹ️  ChromaDB directory does not exist, nothing to clear")
            return True
        
        # Create ChromaDB vectorstore
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(chroma_dir)
        )
        
        # Get collection info before clearing
        collection = vectorstore._collection
        try:
            count = collection.count()
            print(f"📊 Current collection has {count} documents")
        except Exception as e:
            print(f"⚠️  Could not get collection count: {e}")
            count = "unknown"
        
        # Clear the collection
        print("🗑️  Clearing collection...")
        chroma_client = vectorstore._client
        chroma_client.delete_collection(collection_name)
        print(f"✅ Deleted collection: {collection_name}")
        
        # Recreate the collection
        print("🔄 Recreating collection...")
        new_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Recreated collection: {collection_name}")
        
        # Verify the collection is empty
        try:
            new_count = new_collection.count()
            print(f"📊 New collection has {new_count} documents")
        except Exception as e:
            print(f"⚠️  Could not get new collection count: {e}")
        
        print("✅ ChromaDB cleanup completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to clear ChromaDB: {e}")
        return False


def confirm_action(message: str) -> bool:
    """Ask for user confirmation."""
    while True:
        response = input(f"{message} (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clear all data from ChromaDB vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--confirm", 
        action="store_true", 
        help="Skip confirmation prompt"
    )
    
    parser.add_argument(
        "--backup", 
        action="store_true", 
        help="Create a backup before clearing"
    )
    
    parser.add_argument(
        "--collection", 
        default="smart_second_brain",
        help="Collection name to clear (default: smart_second_brain)"
    )
    
    parser.add_argument(
        "--directory", 
        default="./chroma_db",
        help="ChromaDB directory path (default: ./chroma_db)"
    )
    
    args = parser.parse_args()
    
    print("🧠 Smart Second Brain - ChromaDB Cleanup Script")
    print("=" * 50)
    
    # Load environment
    load_environment()
    
    # Check if ChromaDB directory exists
    chroma_dir = Path(args.directory)
    if not chroma_dir.exists():
        print(f"ℹ️  ChromaDB directory does not exist: {chroma_dir}")
        print("Nothing to clear.")
        return 0
    
    # Show current state
    print(f"📁 ChromaDB Directory: {chroma_dir.absolute()}")
    print(f"🗂️  Collection Name: {args.collection}")
    
    # Create backup if requested
    if args.backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = project_root / f"chroma_db_backup_{timestamp}"
        
        if not create_backup(backup_dir):
            print("❌ Backup failed, aborting cleanup")
            return 1
        
        print(f"💾 Backup created: {backup_dir}")
    
    # Confirmation
    if not args.confirm:
        print("\n⚠️  WARNING: This will permanently delete ALL vector data!")
        print("   This operation cannot be undone.")
        
        if not confirm_action("Are you sure you want to clear the ChromaDB?"):
            print("❌ Operation cancelled by user")
            return 0
    
    # Perform cleanup
    print("\n🚀 Starting cleanup...")
    success = clear_chromadb(args.collection, args.directory)
    
    if success:
        print("\n✅ ChromaDB cleanup completed successfully!")
        print("   The vector database is now empty and ready for new data.")
        return 0
    else:
        print("\n❌ ChromaDB cleanup failed!")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
