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
from shared.utils.logging_config import setup_agentic_logging

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
    
    def put(self, *args, **kwargs) -> None:
        """
        Store a checkpoint in Redis.
        
        Args:
            Flexible signature; extracts config, checkpoint, metadata from args/kwargs.
        """
        try:
            # Extract parameters from kwargs or args
            config = kwargs.get("config") if "config" in kwargs else (args[0] if len(args) > 0 else {})
            checkpoint = kwargs.get("checkpoint") if "checkpoint" in kwargs else (args[1] if len(args) > 1 else {})
            metadata = kwargs.get("metadata") if "metadata" in kwargs else (args[2] if len(args) > 2 else {})

            if not isinstance(config, dict):
                config = {}
            if not isinstance(checkpoint, dict):
                checkpoint = {"value": checkpoint}
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}

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
    
    def get_next_version(self, *args, **kwargs) -> str:
        """
        Generate and return the next version identifier for a given config.

        LangGraph expects checkpointers to provide versioning so that new
        checkpoints for the same thread/config can be uniquely identified.

        Returns:
            A monotonically increasing string version identifier scoped to the thread.
        """
        try:
            # Support multiple calling conventions: (config), (config, _), (config=config)
            if "config" in kwargs:
                config = kwargs["config"]
            elif len(args) >= 1 and isinstance(args[0], dict):
                config = args[0]
            else:
                config = {}
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                # Fallback to a global version counter if thread_id is missing
                version_key = "checkpoint:global:version"
            else:
                version_key = f"checkpoint:{thread_id}:version"

            # Use Redis atomic INCR to get the next version number
            next_version_int = self.redis_client.incr(version_key)
            # Set an expiry on the version key to align with TTL policy
            self.redis_client.expire(version_key, self.ttl_seconds)
            return str(next_version_int)
        except redis.RedisError as e:
            logger.error(f"Redis error generating next version: {e}")
            # Fallback to timestamp-based version
            return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        except Exception as e:
            logger.error(f"Unexpected error generating next version: {e}")
            return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

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
    
    def put_writes(self, config: Dict, writes: Any, metadata: Dict) -> None:
        """
        Store multiple writes (batch) for a given config.

        Some LangGraph executors call `put_writes` to persist a set of
        incremental updates as an atomic operation. We serialize and store
        the batch in a single Redis entry keyed by version.

        Args:
            config: Configuration dictionary
            writes: Arbitrary write payload provided by the executor
            metadata: Associated metadata for this batch
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                logger.warning("No thread_id in config, skipping put_writes")
                return

            version = self.get_next_version(config)
            key = f"checkpoint:{thread_id}:writes:{version}"

            payload = {
                "config": config,
                "writes": writes,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat(),
                "version": version,
            }

            self.redis_client.setex(key, self.ttl_seconds, json.dumps(payload))

            # Update latest pointer
            latest_key = self._get_thread_key(thread_id)
            self.redis_client.setex(latest_key, self.ttl_seconds, key)

            logger.debug(f"💾 Stored put_writes batch for thread {thread_id} v{version}")
        except redis.RedisError as e:
            logger.error(f"Redis error in put_writes: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in put_writes: {e}")

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
