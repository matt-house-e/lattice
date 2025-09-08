"""
Checkpoint utilities for the Lattice enrichment tool.

Provides functionality to save and resume enrichment progress,
preventing data loss during long-running enrichment processes.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages checkpointing operations for enrichment processes.
    
    Handles saving and loading checkpoint data to allow resuming
    interrupted enrichment operations.
    """
    
    def __init__(self, config):
        """
        Initialize checkpoint manager.
        
        Args:
            config: EnrichmentConfig instance with checkpoint settings
        """
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
    
    def _get_checkpoint_path(self, base_path: str, category: str) -> Tuple[str, str]:
        """
        Generate checkpoint file paths.
        
        Args:
            base_path: Original data file path or identifier
            category: Field category being processed
            
        Returns:
            Tuple of (data_checkpoint_path, metadata_checkpoint_path)
        """
        if self.checkpoint_dir:
            checkpoint_dir = Path(self.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
        else:
            # Use same directory as base path if it's a file path
            if os.path.exists(os.path.dirname(base_path)) if os.path.dirname(base_path) else True:
                checkpoint_dir = Path(os.path.dirname(base_path)) if os.path.dirname(base_path) else Path.cwd()
            else:
                checkpoint_dir = Path.cwd()
        
        # Create safe filename from base_path and category
        base_name = Path(base_path).stem if os.path.exists(base_path) else str(hash(base_path))
        safe_category = "".join(c for c in category if c.isalnum() or c in ('-', '_'))
        
        data_path = checkpoint_dir / f"{base_name}_{safe_category}_checkpoint.csv"
        metadata_path = checkpoint_dir / f"{base_name}_{safe_category}_checkpoint.json"
        
        return str(data_path), str(metadata_path)
    
    def save_checkpoint(self, 
                       df: pd.DataFrame,
                       base_path: str,
                       category: str,
                       last_processed_idx: int,
                       fields_dict: Dict,
                       overwrite_fields: bool,
                       additional_metadata: Optional[Dict] = None) -> bool:
        """
        Save current enrichment state to checkpoint files.
        
        Args:
            df: Current DataFrame state
            base_path: Original data file path or identifier
            category: Field category being processed
            last_processed_idx: Index of last successfully processed row
            fields_dict: Fields being processed
            overwrite_fields: Whether overwriting existing fields
            additional_metadata: Optional additional metadata to save
            
        Returns:
            bool: True if checkpoint saved successfully
        """
        if not self.config.enable_checkpointing:
            return True
            
        try:
            data_path, metadata_path = self._get_checkpoint_path(base_path, category)
            
            # Save DataFrame
            df.to_csv(data_path, index=False)
            
            # Save metadata
            metadata = {
                'timestamp': time.time(),
                'category': category,
                'last_processed_idx': last_processed_idx,
                'total_rows': len(df),
                'fields_dict': fields_dict,
                'overwrite_fields': overwrite_fields,
                'config': {
                    'batch_size': self.config.batch_size,
                    'max_workers': self.config.max_workers,
                    'row_delay': self.config.row_delay,
                    'checkpoint_interval': self.config.checkpoint_interval
                }
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Checkpoint saved at row {last_processed_idx}: {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, 
                       base_path: str, 
                       category: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Load checkpoint data if available.
        
        Args:
            base_path: Original data file path or identifier
            category: Field category being processed
            
        Returns:
            Tuple of (DataFrame, metadata) if checkpoint found, None otherwise
        """
        if not self.config.enable_checkpointing or not self.config.auto_resume:
            return None
            
        try:
            data_path, metadata_path = self._get_checkpoint_path(base_path, category)
            
            # Check if both checkpoint files exist
            if not os.path.exists(data_path) or not os.path.exists(metadata_path):
                return None
            
            # Load metadata first to validate
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate metadata
            if metadata.get('category') != category:
                logger.warning(f"Checkpoint category mismatch: expected {category}, got {metadata.get('category')}")
                return None
            
            # Load DataFrame
            df = pd.read_csv(data_path)
            
            logger.info(f"Checkpoint loaded: {len(df)} rows, last processed: {metadata.get('last_processed_idx', 0)}")
            return df, metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def cleanup_checkpoints(self, base_path: str, category: str) -> bool:
        """
        Remove checkpoint files after successful completion.
        
        Args:
            base_path: Original data file path or identifier
            category: Field category that was processed
            
        Returns:
            bool: True if cleanup successful
        """
        if not self.config.enable_checkpointing:
            return True
            
        try:
            data_path, metadata_path = self._get_checkpoint_path(base_path, category)
            
            # Remove files if they exist
            for path in [data_path, metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed checkpoint file: {path}")
            
            logger.info(f"Checkpoint files cleaned up for {category}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoint files: {e}")
            return False
    
    def list_checkpoints(self) -> Dict[str, Dict]:
        """
        List available checkpoint files.
        
        Returns:
            Dict mapping checkpoint identifiers to metadata
        """
        checkpoints = {}
        
        if not self.config.enable_checkpointing:
            return checkpoints
            
        checkpoint_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir else Path.cwd()
        
        if not checkpoint_dir.exists():
            return checkpoints
            
        try:
            # Find all checkpoint metadata files
            for metadata_file in checkpoint_dir.glob("*_checkpoint.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if corresponding data file exists
                    data_file = metadata_file.with_suffix('.csv')
                    if data_file.exists():
                        checkpoints[str(metadata_file.stem)] = {
                            'metadata_path': str(metadata_file),
                            'data_path': str(data_file),
                            'category': metadata.get('category'),
                            'last_processed_idx': metadata.get('last_processed_idx'),
                            'total_rows': metadata.get('total_rows'),
                            'timestamp': metadata.get('timestamp')
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint metadata {metadata_file}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            
        return checkpoints