"""
Redis-based Checkpointer for LangGraph Workflows.

This module provides a Redis-based checkpointer that implements LangGraph's
checkpointer interface, enabling persistent conversation state across
server restarts and providing scalable state management.

Key Features:
- Persistent conversation state across server restarts
- Thread-isolated state management
- Redis-based storage for scalability
- Compatible with LangGraph's checkpointer interface
- Automatic expiration of old states
"""

import json
import redis
from typing import Dict, List, Optional, Any, Iterator, Tuple
from datetime import datetime, timedelta
import logging

# Import centralized logging
from ..utils.logging_config import setup_agentic_logging

# Set up logging
logger = setup_agentic_logging()


class RedisCheckpointer:
    """
    Redis-based checkpointer for LangGraph workflows.
    
    This class implements LangGraph's checkpointer interface using Redis
    as the backend storage, providing persistent state management across
    server restarts and enabling scalable multi-user conversations.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 1, ttl_days: int = 7):
        """
        Initialize the Redis checkpointer.
        
        Args:
            redis_url: Redis connection URL
            db: Redis database number (using db=1 to avoid conflicts with conversation memory)
            ttl_days: Time-to-live for states in days
        """
        try:
            self.redis_client = redis.from_url(redis_url, db=db, decode_responses=True)
            self.ttl_seconds = ttl_days * 24 * 60 * 60
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis checkpointer initialized successfully (db={db}, ttl={ttl_days}d)")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis for checkpointer: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing Redis checkpointer: {e}")
            raise
    
    def _get_thread_key(self, thread_id: str) -> str:
        """Generate Redis key for a thread's state."""
        return f"checkpoint:{thread_id}"
    
    def _get_config_key(self, thread_id: str, config: Dict) -> str:
        """Generate Redis key for a specific configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return f"checkpoint:{thread_id}:{hash(config_str)}"
    
    def put(self, config: Dict, checkpoint: Dict, metadata: Dict) -> None:
        """
        Store a checkpoint in Redis.
        
        Args:
            config: Configuration dictionary
            thread_id: Thread identifier
            checkpoint: Checkpoint data
            metadata: Checkpoint metadata
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                logger.warning("No thread_id in config, skipping checkpoint storage")
                return
            
            config_key = self._get_config_key(thread_id, config)
            
            checkpoint_data = {
                "config": config,
                "checkpoint": checkpoint,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store checkpoint
            self.redis_client.setex(
                config_key,
                self.ttl_seconds,
                json.dumps(checkpoint_data)
            )
            
            # Update thread's latest checkpoint reference
            thread_key = self._get_thread_key(thread_id)
            self.redis_client.setex(
                thread_key,
                self.ttl_seconds,
                config_key
            )
            
            logger.debug(f"💾 Stored checkpoint for thread {thread_id}")
            
        except redis.RedisError as e:
            logger.error(f"Redis error storing checkpoint: {e}")
        except Exception as e:
            logger.error(f"Unexpected error storing checkpoint: {e}")
    
    def get_tuple(self, config: Dict) -> Optional[Tuple[Dict, Dict, Dict]]:
        """
        Retrieve a checkpoint from Redis.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (config, checkpoint, metadata) or None if not found
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                logger.warning("No thread_id in config, cannot retrieve checkpoint")
                return None
            
            config_key = self._get_config_key(thread_id, config)
            checkpoint_data = self.redis_client.get(config_key)
            
            if not checkpoint_data:
                logger.debug(f"🔍 No checkpoint found for thread {thread_id}")
                return None
            
            data = json.loads(checkpoint_data)
            logger.debug(f"📖 Retrieved checkpoint for thread {thread_id}")
            
            return (data["config"], data["checkpoint"], data["metadata"])
            
        except redis.RedisError as e:
            logger.error(f"Redis error retrieving checkpoint: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving checkpoint: {e}")
            return None
    
    def list(self, config: Dict, before: Optional[str] = None, limit: Optional[int] = None) -> Iterator[Tuple[Dict, Dict, Dict]]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Configuration dictionary
            before: Optional checkpoint ID to list before
            limit: Optional limit on number of checkpoints
            
        Yields:
            Tuples of (config, checkpoint, metadata)
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                logger.warning("No thread_id in config, cannot list checkpoints")
                return
            
            # Get all checkpoint keys for this thread
            pattern = f"checkpoint:{thread_id}:*"
            checkpoint_keys = self.redis_client.keys(pattern)
            
            if not checkpoint_keys:
                logger.debug(f"🔍 No checkpoints found for thread {thread_id}")
                return
            
            # Sort by timestamp (newest first)
            checkpoints = []
            for key in checkpoint_keys:
                checkpoint_data = self.redis_client.get(key)
                if checkpoint_data:
                    data = json.loads(checkpoint_data)
                    checkpoints.append((data["timestamp"], key, data))
            
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            
            # Apply limit
            if limit:
                checkpoints = checkpoints[:limit]
            
            # Yield checkpoints
            for timestamp, key, data in checkpoints:
                yield (data["config"], data["checkpoint"], data["metadata"])
                
        except redis.RedisError as e:
            logger.error(f"Redis error listing checkpoints: {e}")
        except Exception as e:
            logger.error(f"Unexpected error listing checkpoints: {e}")
    
    def get_tuple_latest(self, config: Dict) -> Optional[Tuple[Dict, Dict, Dict]]:
        """
        Get the latest checkpoint for a thread.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (config, checkpoint, metadata) or None if not found
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                logger.warning("No thread_id in config, cannot get latest checkpoint")
                return None
            
            # Get the latest checkpoint key for this thread
            thread_key = self._get_thread_key(thread_id)
            latest_key = self.redis_client.get(thread_key)
            
            if not latest_key:
                logger.debug(f"🔍 No latest checkpoint found for thread {thread_id}")
                return None
            
            checkpoint_data = self.redis_client.get(latest_key)
            if not checkpoint_data:
                logger.debug(f"🔍 Latest checkpoint data not found for thread {thread_id}")
                return None
            
            data = json.loads(checkpoint_data)
            logger.debug(f"📖 Retrieved latest checkpoint for thread {thread_id}")
            
            return (data["config"], data["checkpoint"], data["metadata"])
            
        except redis.RedisError as e:
            logger.error(f"Redis error getting latest checkpoint: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting latest checkpoint: {e}")
            return None
    
    def delete(self, config: Dict) -> None:
        """
        Delete a checkpoint from Redis.
        
        Args:
            config: Configuration dictionary
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                logger.warning("No thread_id in config, cannot delete checkpoint")
                return
            
            config_key = self._get_config_key(thread_id, config)
            self.redis_client.delete(config_key)
            
            logger.debug(f"🗑️ Deleted checkpoint for thread {thread_id}")
            
        except redis.RedisError as e:
            logger.error(f"Redis error deleting checkpoint: {e}")
        except Exception as e:
            logger.error(f"Unexpected error deleting checkpoint: {e}")
    
    def clear_thread(self, thread_id: str) -> None:
        """
        Clear all checkpoints for a specific thread.
        
        Args:
            thread_id: Thread identifier
        """
        try:
            # Get all checkpoint keys for this thread
            pattern = f"checkpoint:{thread_id}:*"
            checkpoint_keys = self.redis_client.keys(pattern)
            
            if checkpoint_keys:
                self.redis_client.delete(*checkpoint_keys)
                logger.info(f"🗑️ Cleared {len(checkpoint_keys)} checkpoints for thread {thread_id}")
            
            # Clear thread reference
            thread_key = self._get_thread_key(thread_id)
            self.redis_client.delete(thread_key)
            
        except redis.RedisError as e:
            logger.error(f"Redis error clearing thread checkpoints: {e}")
        except Exception as e:
            logger.error(f"Unexpected error clearing thread checkpoints: {e}")
    
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
    
    def get_stats(self) -> Dict:
        """
        Get statistics about stored checkpoints.
        
        Returns:
            Dictionary with checkpoint statistics
        """
        try:
            checkpoint_keys = self.redis_client.keys("checkpoint:*")
            total_checkpoints = len([k for k in checkpoint_keys if ":" in k.split(":")[-1]])
            total_threads = len(set(k.split(":")[1] for k in checkpoint_keys if len(k.split(":")) >= 2))
            
            return {
                "total_checkpoints": total_checkpoints,
                "total_threads": total_threads,
                "redis_info": self.redis_client.info("memory")
            }
        except redis.RedisError as e:
            logger.error(f"Redis error getting stats: {e}")
            return {"error": str(e)}


# Global instance for the application
redis_checkpointer = RedisCheckpointer()
