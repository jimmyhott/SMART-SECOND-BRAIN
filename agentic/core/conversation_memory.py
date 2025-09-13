"""
Redis-based Conversation Memory Manager for Smart Second Brain.

This module provides a Redis-based conversation memory solution to replace
the non-functional LangGraph MemorySaver. It maintains thread-isolated
conversation history using Redis for persistence and scalability.

Key Features:
- Thread-isolated conversation memory
- Persistent conversation history across server restarts
- Redis-based storage for scalability
- Automatic expiration of old conversations
- Compatible with existing KnowledgeState
"""

import json
import redis
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Import centralized logging
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from shared.utils.logging_config import setup_agentic_logging

# Set up logging
logger = setup_agentic_logging()


class RedisConversationMemoryManager:
    """
    Manages conversation memory for thread-isolated conversations using Redis.
    
    This class provides a reliable, persistent alternative to LangGraph's MemorySaver
    for maintaining conversation history across invocations and server restarts.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 0):
        """
        Initialize the Redis conversation memory manager.
        
        Args:
            redis_url: Redis connection URL
            db: Redis database number
        """
        try:
            self.redis_client = redis.from_url(redis_url, db=db, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis conversation memory manager initialized successfully")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing Redis: {e}")
            raise
    
    def _get_thread_key(self, thread_id: str) -> str:
        """Generate Redis key for a thread."""
        return f"conversation:{thread_id}"
    
    def add_message(self, thread_id: str, role: str, content: str) -> None:
        """
        Add a message to a conversation thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            role: Message role ("user" or "assistant")
            content: Message content
        """
        try:
            thread_key = self._get_thread_key(thread_id)
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add message to the end of the list (chronological order)
            self.redis_client.rpush(thread_key, json.dumps(message))
            
            # Set expiration for the thread (7 days)
            self.redis_client.expire(thread_key, 7 * 24 * 60 * 60)
            
            logger.debug(f"💬 Added {role} message to thread {thread_id}: {content[:50]}...")
        except redis.RedisError as e:
            logger.error(f"Redis error adding message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error adding message: {e}")
    
    def get_conversation_history(self, thread_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            List of messages in the format expected by KnowledgeState
        """
        try:
            thread_key = self._get_thread_key(thread_id)
            messages = self.redis_client.lrange(thread_key, 0, -1)
            
            if not messages:
                logger.debug(f"🔍 No conversation history found for thread: {thread_id}")
                return []
            
            # Parse JSON messages and return in chronological order
            conversation_history = []
            for message_json in messages:
                try:
                    message = json.loads(message_json)
                    conversation_history.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message JSON: {message_json}")
                    continue
            
            logger.debug(f"📜 Retrieved {len(conversation_history)} messages for thread: {thread_id}")
            return conversation_history
        except redis.RedisError as e:
            logger.error(f"Redis error retrieving conversation history: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving conversation history: {e}")
            return []
    
    def get_conversation_summary(self, thread_id: str) -> str:
        """
        Get a formatted conversation summary for a thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            Formatted conversation history as a string
        """
        messages = self.get_conversation_history(thread_id)
        if not messages:
            return ""
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(formatted_messages)
    
    def clear_thread(self, thread_id: str) -> None:
        """
        Clear the conversation history for a thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
        """
        try:
            thread_key = self._get_thread_key(thread_id)
            self.redis_client.delete(thread_key)
            logger.info(f"🗑️ Cleared conversation history for thread: {thread_id}")
        except redis.RedisError as e:
            logger.error(f"Redis error clearing conversation history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error clearing conversation history: {e}")
    
    def get_thread_info(self, thread_id: str) -> Optional[Dict]:
        """
        Get information about a conversation thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            Dictionary with thread information or None if thread doesn't exist
        """
        try:
            thread_key = self._get_thread_key(thread_id)
            messages = self.redis_client.lrange(thread_key, 0, -1)
            
            if not messages:
                return None
            
            # Get first and last message timestamps
            first_message = json.loads(messages[0])
            last_message = json.loads(messages[-1])
            
            return {
                "thread_id": thread_id,
                "message_count": len(messages),
                "created_at": first_message.get("timestamp", ""),
                "last_updated": last_message.get("timestamp", "")
            }
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting thread info: {e}")
            return None
    
    def list_threads(self) -> List[str]:
        """
        Get a list of all active thread IDs.
        
        Returns:
            List of thread IDs
        """
        try:
            conversation_keys = self.redis_client.keys("conversation:*")
            thread_ids = [key.replace("conversation:", "") for key in conversation_keys]
            return thread_ids
        except redis.RedisError as e:
            logger.error(f"Redis error listing threads: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the conversation memory.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            conversation_keys = self.redis_client.keys("conversation:*")
            total_threads = len(conversation_keys)
            total_messages = 0
            
            for key in conversation_keys:
                total_messages += self.redis_client.llen(key)
            
            return {
                "total_threads": total_threads,
                "total_messages": total_messages,
                "threads": [key.replace("conversation:", "") for key in conversation_keys],
                "redis_info": self.redis_client.info("memory")
            }
        except redis.RedisError as e:
            logger.error(f"Redis error getting stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            True if Redis is accessible, False otherwise.
        """
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    def clear_all_history(self) -> None:
        """Clear all conversation history across all threads."""
        try:
            # Find all conversation keys
            conversation_keys = self.redis_client.keys("conversation:*")
            if conversation_keys:
                self.redis_client.delete(*conversation_keys)
                logger.info(f"🗑️ Cleared {len(conversation_keys)} conversation threads")
            else:
                logger.info("No conversation threads found to clear")
        except redis.RedisError as e:
            logger.error(f"Redis error clearing all history: {e}")
        except Exception as e:
            logger.error(f"Unexpected error clearing all history: {e}")


# Global instance for the application
conversation_memory = RedisConversationMemoryManager()