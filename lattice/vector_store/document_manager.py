from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import os
from .document_processor import DocumentProcessor, DocumentProcessorConfig
from .vector_store import VectorStore, VectorStoreConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DocumentManagerConfig:
    """Configuration for document management."""
    batch_size: int = 50
    parallel_processing: bool = False
    max_workers: int = 4
    skip_existing: bool = True
    doc_processor_config: Optional[DocumentProcessorConfig] = None
    vector_store_config: Optional[VectorStoreConfig] = None
    supported_extensions: Set[str] = field(default_factory=lambda: {'.pdf', '.html', '.htm'})

class DocumentManager:
    """
    Base class for managing document processing and storage in vector store.
    Provides core functionality for document handling that can be extended
    for specific use cases.
    """
    def __init__(
        self,
        vector_store: VectorStore,
        config: Optional[DocumentManagerConfig] = None,
        document_processor: Optional[DocumentProcessor] = None
    ):
        """
        Initialize document manager.
        
        Args:
            vector_store: VectorStore instance for document storage (required)
            config: Optional configuration settings
            document_processor: Optional custom document processor
        """
        if not vector_store:
            raise ValueError("vector_store is required")
            
        self.config = config or DocumentManagerConfig()
        self.vector_store = vector_store
        self.document_processor = document_processor or DocumentProcessor(
            config=self.config.doc_processor_config
        )

    def _normalize_path(self, path: str) -> str:
        """Normalize path separators for consistent comparison."""
        return os.path.normpath(path).replace('\\', '/')

    def is_document_processed(self, source_path: str, filter_dict: Optional[Dict] = None) -> bool:
        """
        Check if a document is already processed by looking up its metadata and verifying content.
        
        Args:
            source_path: Path to the document file
            filter_dict: Additional metadata filters to apply
            
        Returns:
            bool: True if document exists in vector store with content
        """
        if not self.config.skip_existing:
            return False
            
        # Normalize path for comparison
        normalized_path = self._normalize_path(source_path)
        logger.debug(f"Checking for existing document: {normalized_path}")
        
        # Prepare search filters
        search_filters = filter_dict.copy() if filter_dict else {}
        search_filters['source_path'] = normalized_path
        
        # Search with empty query but strict metadata matching
        results = self.vector_store.search(
            " ",
            filter_dict=search_filters,
            limit=1
        )
        
        # Verify that we found a document AND it has content
        if results and results[0].get('content', '').strip():
            logger.info(f"Found existing document with content: {normalized_path}")
            return True
                
        return False

    def delete_documents(self, filter_dict: Dict) -> bool:
        """
        Delete documents from vector store based on metadata filters.
        
        Args:
            filter_dict: Dictionary of metadata filters (e.g., {'uuid': '123', 'report_name': 'Report'})
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.vector_store.delete_documents(filter_dict)

    def process_documents(
        self,
        file_paths: List[str],
        base_metadata: Optional[Dict] = None,
        filter_dict: Optional[Dict] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Process a list of documents and add them to the vector store.
        
        Args:
            file_paths: List of paths to documents to process
            base_metadata: Optional metadata to include with all documents
            filter_dict: Optional filter for checking existing documents
            
        Returns:
            Dict containing processing statistics
        """
        stats = {
            'total': len(file_paths),
            'processed': 0,
            'added': 0,
            'skipped': 0,
            'error': 0,
            'unsupported': 0
        }
        
        base_metadata = base_metadata or {}
        
        # Filter to supported file types
        supported_files = [
            f for f in file_paths 
            if os.path.splitext(f)[1].lower() in self.config.supported_extensions
        ]
        stats['unsupported'] = len(file_paths) - len(supported_files)
        
        for file_path in supported_files:
            try:
                # Check if document already processed
                if self.is_document_processed(file_path, filter_dict=filter_dict):
                    logger.info(f"Skipping existing document: {file_path}")
                    stats['skipped'] += 1
                    continue

                # Process document
                documents = self.document_processor.process_document(file_path, base_metadata)
                
                if not documents:
                    logger.warning(f"No content extracted from document: {file_path}")
                    stats['error'] += 1
                    continue
                
                # Format documents for vector store
                formatted_docs = [{
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in documents]
                
                # Add to vector store in batches
                if formatted_docs:
                    success = self.vector_store.add_documents(
                        formatted_docs,
                        batch_size=self.config.batch_size
                    )
                    if not success:
                        raise Exception("Failed to add documents to vector store")
                    
                    stats['added'] += len(formatted_docs)
                    stats['processed'] += 1
                    
                    logger.info(f"Successfully processed document: {file_path}")
                    logger.info(f"Added {len(formatted_docs)} chunks to vector store")
                
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {str(e)}")
                stats['error'] += 1
        
        return stats

    def search_documents(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        limit: int = 20,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for documents in the vector store.
        
        Args:
            query: Search query string
            filter_dict: Optional metadata filters
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matching documents with scores
        """
        return self.vector_store.search(
            query=query,
            filter_dict=filter_dict,
            limit=limit,
            min_score=min_score
        )
