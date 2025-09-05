import os
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from langchain.schema import Document
from backend.table_enrichment_tool.logger import configure_logger
import pdfplumber
from bs4 import BeautifulSoup
import tiktoken

logger = configure_logger(__name__)

@dataclass
class DocumentProcessorConfig:
    """Configuration for document processing."""
    min_tokens: int = 500
    max_tokens: int = 2000
    overlap_pct: float = 0.1  # Reduced overlap to 10%
    supported_extensions: Set[str] = field(default_factory=lambda: {'.pdf', '.html', '.htm'})
    tokenizer_name: str = "cl100k_base"
    extract_headers: bool = True  # For HTML processing
    header_tags: Set[str] = field(default_factory=lambda: {'h1', 'h2'})
    skip_elements: Set[str] = field(default_factory=lambda: {'script', 'style'})
    min_chunk_similarity: float = 0.9  # Threshold for considering chunks duplicates

class DocumentProcessor:
    """Processes documents into chunks suitable for vector storage."""
    
    def __init__(self, config: Optional[DocumentProcessorConfig] = None):
        self.config = config or DocumentProcessorConfig()
        self.tokenizer = tiktoken.get_encoding(self.config.tokenizer_name)
        self._chunk_cache = set()  # Cache for deduplication
        
    def process_document(self, file_path: str, metadata: Dict) -> List[Document]:
        """Process document based on file type."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in self.config.supported_extensions:
            logger.warning(f"Unsupported file format: {file_extension}")
            return []
            
        try:
            # Normalize path for consistent storage
            normalized_path = os.path.normpath(file_path).replace('\\', '/')
            
            # Create new metadata dict while preserving existing UUID
            new_metadata = metadata.copy()
            
            # Add path information
            new_metadata.update({
                'source_path': normalized_path,  # New key
            })
            
            # Only try to extract UUID from path if not already present
            if 'uuid' not in new_metadata:
                logger.debug(f"UUID not found in metadata for {file_path}, attempting to extract from path")
                try:
                    # Assuming path structure includes UUID directory
                    path_parts = normalized_path.split(os.sep)
                    for part in path_parts:
                        if len(part) == 36 and part.count('-') == 4:  # UUID format check
                            new_metadata['uuid'] = part
                            logger.debug(f"Extracted UUID from path: {part}")
                            break
                except Exception as e:
                    logger.warning(f"Could not extract uuid from path: {e}")
            
            if 'uuid' not in new_metadata:
                logger.warning(f"No uuid found in metadata or path for document: {file_path}")
            
            if file_extension == '.pdf':
                return self._process_pdf(file_path, new_metadata)
            elif file_extension in {'.html', '.htm'}:
                return self._process_html(file_path, new_metadata)
            return []
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return []
    
    def _process_pdf(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """Process a PDF file into chunks with metadata."""
        chunks = []
        current_chunk_num = 0
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    # Update metadata with page-specific information
                    metadata = base_metadata.copy()
                    metadata.update({
                        'page_number': page_num,
                        'total_pages': len(pdf.pages),
                        'file_type': 'pdf'
                    })
                    
                    # Chunk the page content
                    page_chunks, current_chunk_num = self._chunk_text(text, metadata, current_chunk_num)
                    chunks.extend(page_chunks)
        return chunks
    
    def _process_html(self, file_path: str, base_metadata: Dict) -> List[Document]:
        """Process an HTML file into chunks with metadata."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            
        # Remove unwanted elements
        for element in soup(self.config.skip_elements):
            element.decompose()
            
        chunks = []
        section_num = 0
        current_chunk_num = 0
        
        if self.config.extract_headers:
            # Process by sections using headers
            for header in soup.find_all(list(self.config.header_tags)):
                section_num += 1
                section_title = header.get_text(strip=True)
                
                # Collect content until next header
                content = []
                for sibling in header.find_next_siblings():
                    if sibling.name in self.config.header_tags:
                        break
                    text = sibling.get_text(separator=' ', strip=True)
                    if text:
                        content.append(text)
                
                if content:
                    # Update metadata with section information
                    metadata = base_metadata.copy()
                    metadata.update({
                        'section_number': section_num,
                        'section_title': section_title,
                        'file_type': 'html'
                    })
                    
                    # Chunk the section content
                    section_text = ' '.join(content)
                    section_chunks, current_chunk_num = self._chunk_text(section_text, metadata, current_chunk_num)
                    chunks.extend(section_chunks)
        
        # If no sections found or headers not extracted, process entire content
        if not chunks:
            metadata = base_metadata.copy()
            metadata.update({
                'file_type': 'html'
            })
            text = soup.get_text(separator='\n', strip=True)
            if text:
                all_chunks, _ = self._chunk_text(text, metadata, 0)
                chunks.extend(all_chunks)
        
        return chunks
    
    def _is_duplicate_chunk(self, chunk_content: str) -> bool:
        """Check if chunk content is similar to any existing chunks."""
        # Simple exact match check for now
        return chunk_content in self._chunk_cache

    def _chunk_text(self, text: str, metadata: Dict, start_chunk_num: int = 0) -> Tuple[List[Document], int]:
        """Chunk text content into smaller pieces while preserving metadata and removing duplicates."""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_tokens = 0
        chunk_num = start_chunk_num
        
        # Calculate overlap size in tokens (reduced from previous 20%)
        overlap_tokens = int(self.config.min_tokens * self.config.overlap_pct)
        
        for line in lines:
            line_tokens = len(self.tokenizer.encode(line))
            
            # If adding this line would exceed max tokens, create a new chunk
            if current_tokens + line_tokens > self.config.max_tokens and current_tokens >= self.config.min_tokens:
                chunk_content = '\n'.join(current_chunk)
                
                # Only add chunk if it's not a duplicate
                if not self._is_duplicate_chunk(chunk_content):
                    chunk_num += 1
                    # Update metadata with chunk information
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_number': chunk_num,
                        'token_count': current_tokens,
                        'chunk_id': f"{metadata.get('source_path', 'unknown')}#chunk{chunk_num}"
                    })
                    
                    chunks.append(Document(
                        page_content=chunk_content,
                        metadata=chunk_metadata
                    ))
                    self._chunk_cache.add(chunk_content)
                
                # Keep minimal overlap for next chunk
                overlap_start = max(0, len(current_chunk) - int(len(current_chunk) * self.config.overlap_pct))
                current_chunk = current_chunk[overlap_start:]
                current_tokens = sum(len(self.tokenizer.encode(l)) for l in current_chunk)
            
            current_chunk.append(line)
            current_tokens += line_tokens
        
        # Handle the last chunk if it meets minimum token requirement
        if current_chunk and current_tokens >= self.config.min_tokens:
            chunk_content = '\n'.join(current_chunk)
            if not self._is_duplicate_chunk(chunk_content):
                chunk_num += 1
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_number': chunk_num,
                    'token_count': current_tokens,
                    'chunk_id': f"{metadata.get('source_path', 'unknown')}#chunk{chunk_num}"
                })
                
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                ))
                self._chunk_cache.add(chunk_content)
        
        return chunks, chunk_num
