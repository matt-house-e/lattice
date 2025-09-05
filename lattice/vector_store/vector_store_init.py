"""
Vector store initialization and management module.
Provides a singleton-like interface for vector store access.
"""

from typing import Optional
from .vector_store import VectorStore, VectorStoreConfig
from backend.system.logger import configure_logger
import atexit
import threading

# Config Parameters
from utils.config_loader import get_value
VECTOR_STORE_PATH = get_value("VECTOR_STORE_PATH", default="data/vector_store")

logger = configure_logger(__name__)

class VectorStoreManager:
    """
    Manager class for vector store initialization and access.
    Provides a singleton-like interface to ensure only one instance exists.
    """
    _instance: Optional[VectorStore] = None
    _lock = threading.Lock()
    _initialized = False
    
    @classmethod
    def initialize(cls, store_path: Optional[str] = None) -> None:
        """
        Initialize the vector store with the given configuration.
        
        Args:
            store_path: Optional path to vector store directory
                       (defaults to VECTOR_STORE_PATH from config)
        """
        with cls._lock:
            if cls._initialized:
                return
                
            config = VectorStoreConfig(
                store_path=store_path or VECTOR_STORE_PATH
            )
            
            if cls._instance is None:
                logger.info(f"Initializing vector store at: {config.store_path}")
                cls._instance = VectorStore(config=config)
                cls._initialized = True
            
    @classmethod
    def get_instance(cls) -> VectorStore:
        """
        Get the vector store instance, initializing it if necessary.
        
        Returns:
            The VectorStore instance
        """
        if not cls._initialized:
            cls.initialize()
        return cls._instance
        
    @classmethod
    def close(cls) -> None:
        """
        Clean up vector store resources.
        """
        with cls._lock:
            if cls._initialized:
                logger.info("Vector store connection closed")
                cls._instance = None
                cls._initialized = False

# Create global instance
vector_store = VectorStoreManager()

# Register cleanup on exit
def cleanup():
    vector_store.close()

atexit.register(cleanup)
