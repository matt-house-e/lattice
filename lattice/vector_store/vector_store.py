from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import json
import time
import numpy as np
from dataclasses import dataclass
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from backend.table_enrichment_tool.logger import configure_logger

logger = configure_logger(__name__)

# Config Parameters
from utils.config_loader import get_value
OPENAI_API_KEY = get_value("OPENAI_API_KEY", default="")
if not OPENAI_API_KEY:
    logger.error("OpenAI API key is not set. Please check your Settings page or config.yaml file.")
    raise ValueError("OpenAI API key is not set. Please check your Settings page or config.yaml file.")

@dataclass
class VectorStoreConfig:
    """Configuration for vector store initialization."""
    store_path: str = "./vector_store"  # Default to current directory
    batch_size: int = 100
    embedding_model: str = "text-embedding-ada-002"
    metric: str = "cosine"  # FAISS metric type
    nprobe: int = 8  # Number of clusters to probe

class VectorStore:
    """
    Vector store for document storage and retrieval.
    Uses FAISS for storage and OpenAI embeddings for vectorization.
    Maintains document uniqueness and handles proper deletion.
    """
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize the vector store with the given configuration."""
        self.config = config or VectorStoreConfig()
        self.store = None
        self.processed_docs = {}
        
        # Ensure store path is absolute and exists
        self.store_path = Path(self.config.store_path).resolve()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set processed docs path
        self.processed_docs_path = self.store_path.parent / "processed_docs.json"
        
        # Initialize embeddings
        logger.info(f"Initializing vector store at: {self.store_path}")
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            api_key=OPENAI_API_KEY
        )
        
        # Initialize store and processed docs
        self._initialize()
        
    def _initialize(self):
        """Initialize store and processed docs in the correct order."""
        # Load or create store first
        self.store = self._load_or_create()
        
        # Then load processed docs and clean up if needed
        if self.processed_docs_path.exists():
            try:
                with open(self.processed_docs_path, 'r') as f:
                    self.processed_docs = json.load(f)
                
                # Clean up stale entries if store has documents
                if hasattr(self.store, 'docstore'):
                    current_uuids = {
                        doc.metadata.get('uuid') 
                        for doc in self.store.docstore._dict.values()
                        if doc.metadata.get('uuid')
                    }
                    
                    # Keep only entries that exist in store
                    self.processed_docs = {
                        doc_id: info for doc_id, info in self.processed_docs.items()
                        if info.get('uuid') in current_uuids
                    }
                    
                    # Save cleaned up version
                    with open(self.processed_docs_path, 'w') as f:
                        json.dump(self.processed_docs, f)
                    
                    logger.info(f"Cleaned up processed docs tracking, kept {len(self.processed_docs)} entries")
                    
            except Exception as e:
                logger.warning(f"Error loading processed docs, starting fresh: {e}")
                self.processed_docs = {}

    def _load_processed_docs(self) -> Dict[str, Dict]:
        """
        Load or create processed documents tracking.
        If store exists, clean up stale entries that no longer exist in the vector store.
        """
        try:
            # If no processed docs file exists, start fresh
            if not self.processed_docs_path.exists():
                return {}
                
            # Load existing processed docs
            with open(self.processed_docs_path, 'r') as f:
                processed_docs = json.load(f)
            
            # If store isn't initialized yet, return without cleanup
            if not hasattr(self, 'store') or self.store is None:
                logger.debug("Store not initialized, skipping processed docs cleanup")
                return processed_docs
            
            # Clean up stale entries if store exists and has documents
            if hasattr(self.store, 'docstore'):
                current_uuids = {
                    doc.metadata.get('uuid') 
                    for doc in self.store.docstore._dict.values()
                    if doc.metadata.get('uuid')
                }
                
                # Keep only entries that exist in store
                processed_docs = {
                    doc_id: info for doc_id, info in processed_docs.items()
                    if info.get('uuid') in current_uuids
                }
                
                # Save cleaned up version
                with open(self.processed_docs_path, 'w') as f:
                    json.dump(processed_docs, f)
                
                logger.info(f"Cleaned up processed docs tracking, kept {len(processed_docs)} entries")
            
            return processed_docs
                
        except Exception as e:
            logger.warning(f"Error loading processed docs, starting fresh: {e}")
            return {}

    def _save_processed_docs(self):
        """Save processed documents tracking."""
        try:
            with open(self.processed_docs_path, 'w') as f:
                json.dump(self.processed_docs, f)
        except Exception as e:
            logger.error(f"Error saving processed docs: {e}")

    def _get_doc_id(self, metadata: Dict, content: Optional[str] = None) -> Optional[str]:
        """
        Generate unique document ID from metadata and optionally content.
        Includes content hash if provided to detect content changes.
        """
        base_id = None
        if 'source_path' in metadata:
            base_id = f"path:{self._normalize_path(metadata['source_path'])}"
        elif 'uuid' in metadata:
            base_id = f"uuid:{metadata['uuid']}"
        
        if base_id and content:
            # Include content hash to detect changes
            import hashlib
            content_hash = hashlib.md5(content.encode()).hexdigest()
            return f"{base_id}:{content_hash}"
            
        return base_id
        
    def _normalize_path(self, path: str) -> str:
        """Normalize path separators for consistent comparison."""
        return os.path.normpath(path).replace('\\', '/')

    def _verify_store(self, store: Optional[FAISS]) -> bool:
        """Verify store is loaded correctly and contains documents."""
        if store and hasattr(store, 'docstore') and hasattr(store.docstore, '_dict'):
            doc_count = len(store.docstore._dict)
            logger.info(f"Successfully loaded vector store with {doc_count} documents")
            return True
        return False

    def _create_empty_store(self) -> FAISS:
        """Create a new empty vector store."""
        # Create a temporary document to initialize the store
        temp_doc = Document(
            page_content="temporary document for initialization",
            metadata={"temp": True}
        )
        
        # Create store with temporary document
        store = FAISS.from_documents(
            [temp_doc],
            self.embeddings,
            distance_strategy=self.config.metric
        )
        
        # Configure FAISS index parameters
        if hasattr(store, 'index'):
            store.index.nprobe = self.config.nprobe
            
        # Remove temporary document
        if hasattr(store, 'docstore') and hasattr(store.docstore, '_dict'):
            store.docstore._dict.clear()
            
        return store

    def _load_or_create(self) -> FAISS:
        """Load existing vector store or create new empty one."""
        try:
            if self.store_path.exists():
                logger.info(f"Loading existing vector store from {self.store_path}")
                store = FAISS.load_local(
                    str(self.store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Only use with trusted data
                )
                if self._verify_store(store):
                    # Configure FAISS index parameters
                    if hasattr(store, 'index'):
                        store.index.nprobe = self.config.nprobe
                    return store
                logger.warning("Vector store loaded but appears to be empty or invalid")
            
            # Create new empty store
            logger.info("Creating new empty vector store")
            store = self._create_empty_store()
            
            # Save to disk
            os.makedirs(self.store_path.parent, exist_ok=True)
            store.save_local(str(self.store_path))
            logger.info(f"Successfully saved empty vector store to {self.store_path}")
            
            return store
            
        except Exception as e:
            logger.error(f"Error loading/creating vector store: {e}", exc_info=True)
            # Create new empty store as fallback
            logger.info("Creating new empty vector store as fallback")
            return self._create_empty_store()

    def _prepare_documents(self, documents: List[Dict[str, str]]) -> List[Document]:
        """Convert raw documents to LangChain Document objects."""
        docs = []
        for doc in documents:
            metadata = doc['metadata'].copy()
            if 'source_path' in metadata:
                metadata['source_path'] = self._normalize_path(metadata['source_path'])
            docs.append(Document(
                page_content=doc['content'],
                metadata=metadata
            ))
        return docs

    def _should_process_document(self, doc: Dict[str, str]) -> bool:
        """Check if document should be processed based on ID and content."""
        doc_id = self._get_doc_id(doc['metadata'], doc['content'])
        if not doc_id:
            logger.warning(f"Document missing identifier in metadata: {doc['metadata']}")
            return False
            
        if doc_id in self.processed_docs:
            logger.debug(f"Skipping already processed document: {doc_id}")
            return False
            
        return True

    def add_documents(self, documents: List[Dict[str, str]], batch_size: Optional[int] = None, force_update: bool = False) -> bool:
        """
        Add documents to vector store, skipping already processed ones.
        
        Args:
            documents: List of dictionaries containing 'content' and 'metadata'
            batch_size: Optional override for batch size (uses config value if not provided)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Filter and track documents to process
            new_docs = []
            for doc in documents:
                doc_id = self._get_doc_id(doc['metadata'], doc['content'])
                if not doc_id:
                    continue
                
                if not force_update and doc_id in self.processed_docs:
                    continue
                
                # If forcing update, delete old version first
                if force_update and doc_id in self.processed_docs:
                    old_id = doc_id.split(':')[0]  # Get base ID without content hash
                    self.delete_documents({k: v for k, v in doc['metadata'].items() 
                                        if k in ['uuid', 'source_path']})
                
                new_docs.append(doc)
                # Only store timestamp and minimal metadata for tracking
                self.processed_docs[doc_id] = {
                    'timestamp': str(time.time()),
                    'uuid': doc['metadata'].get('uuid'),
                    'source_path': doc['metadata'].get('source_path')
                }

            if not new_docs:
                logger.info("No new documents to process")
                self._save_processed_docs()
                return True

            # Process new documents
            docs = self._prepare_documents(new_docs)
            batch_size = batch_size or self.config.batch_size
            
            # Process in batches
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                
                if self.store is None:
                    self.store = FAISS.from_documents(
                        batch, 
                        self.embeddings,
                        distance_strategy=self.config.metric
                    )
                    if hasattr(self.store, 'index'):
                        self.store.index.nprobe = self.config.nprobe
                else:
                    self.store.add_documents(batch)
                
                logger.info(f"Processed batch of {len(batch)} documents")
            
            # Save everything
            os.makedirs(self.store_path.parent, exist_ok=True)
            self.store.save_local(str(self.store_path))
            self._save_processed_docs()
            
            logger.info(f"Successfully processed {len(new_docs)} new documents")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}", exc_info=True)
            return False

    def delete_documents(self, filter_dict: Dict) -> bool:
        """
        Delete documents from vector store based on metadata filters.
        Always rebuilds the index to maintain consistency between FAISS and docstore.
        
        Args:
            filter_dict: Dictionary of metadata filters (e.g., {'uuid': '123'})
            
        Returns:
            bool: True if successful, False only on error
        """
        if self.store is None:
            logger.warning("No vector store available")
            return False

        try:
            # Identify documents to keep
            docs_to_keep = []
            docs_to_delete = []
            
            for doc_id, doc in self.store.docstore._dict.items():
                matches = all(
                    doc.metadata.get(key) == value 
                    for key, value in filter_dict.items()
                )
                if matches:
                    docs_to_delete.append(doc_id)
                    # Remove from processed docs tracking
                    doc_tracking_id = self._get_doc_id(doc.metadata)
                    if doc_tracking_id and doc_tracking_id in self.processed_docs:
                        del self.processed_docs[doc_tracking_id]
                else:
                    docs_to_keep.append(doc)
            
            if not docs_to_delete:
                logger.debug(f"No documents found matching filters: {filter_dict}")
                return True

            # Always rebuild index to maintain consistency
            if docs_to_keep:
                # Create new store with kept documents
                self.store = FAISS.from_documents(
                    docs_to_keep,
                    self.embeddings,
                    distance_strategy=self.config.metric
                )
                if hasattr(self.store, 'index'):
                    self.store.index.nprobe = self.config.nprobe
            else:
                # If no documents left, create empty store
                self.store = self._create_empty_store()
            
            # Save everything
            self.store.save_local(str(self.store_path))
            self._save_processed_docs()
            
            logger.info(f"Successfully deleted {len(docs_to_delete)} documents matching filters: {filter_dict}")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {e}", exc_info=True)
            return False

    def search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        limit: int = 20,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar documents with metadata filtering.
        First filters by metadata, then performs similarity search within filtered subset.
        
        Args:
            query: Search query string
            filter_dict: Dictionary of metadata filters (e.g., {'company_uuid': '123'})
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0-1) for results
            
        Returns:
            List of documents with their content, metadata, and similarity scores
        """
        if self.store is None:
            logger.warning("No vector store available, returning empty results")
            return []

        try:
            # Always apply metadata filters first
            if filter_dict:
                # Get all documents that match the metadata filters
                filtered_docs = []
                for doc in self.store.docstore._dict.values():
                    matches = all(
                        doc.metadata.get(key) == value 
                        for key, value in filter_dict.items()
                    )
                    if matches:
                        filtered_docs.append(doc)
                
                if not filtered_docs:
                    logger.debug(f"No documents found matching filters: {filter_dict}")
                    return []
                
                # Create temporary store with filtered documents for similarity search
                filtered_store = FAISS.from_documents(
                    filtered_docs,
                    self.embeddings,
                    distance_strategy=self.config.metric
                )
                
                # Perform similarity search on filtered subset
                if query and not query.isspace():
                    results = filtered_store.similarity_search_with_score(
                        query,
                        k=limit
                    )
                else:
                    # If no query, return all filtered documents
                    return [{
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': 1.0
                    } for doc in filtered_docs[:limit]]
            else:
                # Check if store is empty
                if not self.store.docstore._dict:
                    logger.debug("Vector store is empty")
                    return []
                    
                # No filters, perform similarity search on entire store
                results = self.store.similarity_search_with_score(
                    query or " ",  # Use space if empty query
                    k=limit
                )
            
            # Format and filter results
            formatted_results = []
            for doc, score in results:
                similarity = 1 - score  # Convert distance to similarity
                if similarity >= min_score:
                    formatted_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []
