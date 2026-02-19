"""
Field management for the Lattice enrichment tool.

Manages field categories, their specifications, and examples.
Handles loading, parsing, and accessing field definitions.

This is a cleaned up version of the original field_manager.py with:
- Cleaner imports (no try/except blocks)
- Better type hints
- Simplified logging
- Enhanced error messages
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import os
from pathlib import Path

from ..core.exceptions import FieldValidationError

logger = logging.getLogger(__name__)


class FieldManager:
    """
    Manages field categories, their specifications, and examples.
    
    Handles loading, parsing, and accessing field definitions from CSV files.
    Validates field definitions and provides structured access to field data.
    """
    
    REQUIRED_COLUMNS = ['Category', 'Field', 'Prompt', 'Instructions', 'Data_Type']
    
    def __init__(self, fields_categories_path: str) -> None:
        """
        Initialize the FieldManager with field category definitions.
        
        Args:
            fields_categories_path: Path to CSV file containing field categories and specifications
            
        Raises:
            FieldValidationError: If the CSV file is invalid or missing required columns
        """
        self.fields_categories_path = Path(fields_categories_path)
        self.categories = self._load_field_categories(fields_categories_path)

    @classmethod
    def from_csv(cls, csv_path: str) -> 'FieldManager':
        """
        Factory method to create FieldManager from CSV file.
        
        Args:
            csv_path: Path to the field categories CSV file
            
        Returns:
            Configured FieldManager instance
        """
        return cls(csv_path)

    def _load_field_categories(self, csv_path: str) -> Dict:
        """
        Load and parse field categories from CSV into structured dictionary.
        
        Args:
            csv_path: Path to CSV file containing field category definitions
            
        Returns:
            Dictionary mapping categories to their field specifications
            
        Raises:
            FieldValidationError: If the CSV file is invalid, missing, or has incorrect format
        """
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FieldValidationError(f"Fields categories CSV file not found at: {csv_path}")
            
            # Try to read the CSV file
            df = pd.read_csv(csv_path)
            
            # Check if DataFrame is empty
            if df.empty:
                raise FieldValidationError(f"Fields categories CSV file is empty: {csv_path}")
            
            # Validate required columns
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise FieldValidationError(
                    f"Fields categories CSV missing required columns: {', '.join(missing_cols)}\n"
                    f"Required columns: {', '.join(self.REQUIRED_COLUMNS)}\n"
                    f"Found columns: {', '.join(df.columns)}"
                )
            
            categories = {}
            
            # Process each row in the CSV
            for idx, row in df.iterrows():
                category = row['Category']
                field = row['Field']
                
                # Validate required fields are not empty
                if pd.isna(category) or pd.isna(field):
                    logger.warning(f"Skipping row {idx} with empty Category or Field: {row.to_dict()}")
                    continue
                
                # Create field specification dictionary
                field_dict = {
                    "prompt": row['Prompt'] if pd.notna(row['Prompt']) else "",
                    "instructions": row['Instructions'] if pd.notna(row['Instructions']) else "",
                    "type": row['Data_Type'] if pd.notna(row['Data_Type']) else "String"
                }
                
                # Add examples if present (use name-based exclusion, not position)
                KNOWN_COLUMNS = {
                    'Category', 'Field', 'Prompt', 'Data_Type', 'Instructions',
                    'Output_Format', 'Quality_Rules', 'Sources',
                    'Good_Example', 'Bad_Example', 'Fallback',
                }
                examples = {}
                for i, col in enumerate(
                    (c for c in df.columns if c not in KNOWN_COLUMNS), start=1
                ):
                    if pd.notna(row[col]):
                        examples[f"example_{i}"] = row[col]
                if examples:
                    field_dict["examples"] = examples
                
                # Add to categories dictionary
                if category not in categories:
                    categories[category] = {}
                categories[category][field] = field_dict
            
            # Validate that at least one category was loaded
            if not categories:
                raise FieldValidationError(f"No valid categories found in CSV file: {csv_path}")
            
            logger.info(f"Successfully loaded {len(categories)} categories from {csv_path}")
            for category_name, fields in categories.items():
                logger.debug(f"Category '{category_name}': {len(fields)} fields")
            
            return categories
            
        except pd.errors.EmptyDataError:
            raise FieldValidationError(f"CSV file is empty or has no data: {csv_path}")
        except pd.errors.ParserError as e:
            raise FieldValidationError(f"Invalid CSV format in {csv_path}: {str(e)}")
        except Exception as e:
            if isinstance(e, FieldValidationError):
                raise  # Re-raise our custom exceptions
            raise FieldValidationError(f"Failed to load field categories CSV {csv_path}: {str(e)}")

    def get_category_fields(self, category: str) -> Dict:
        """
        Get field specifications for a specific category.
        
        Args:
            category: Category name to get fields for
            
        Returns:
            Dictionary of fields and their specifications for the category
            
        Raises:
            FieldValidationError: If the category doesn't exist
        """
        if category not in self.categories:
            available_categories = ", ".join(self.categories.keys())
            raise FieldValidationError(
                f"Category '{category}' not found. Available categories: {available_categories}"
            )
            
        category_fields = self.categories[category]
        return {
            field: {
                "prompt": details["prompt"],
                "type": details["type"],
                "instructions": details["instructions"]
            }
            for field, details in category_fields.items()
        }

    def get_categories(self) -> List[str]:
        """
        Get list of all available categories.
        
        Returns:
            List of category names
        """
        return list(self.categories.keys())
        
    def validate_category(self, category: str) -> bool:
        """
        Check if a category exists.
        
        Args:
            category: Category name to validate
            
        Returns:
            True if category exists, False otherwise
        """
        return category in self.categories
    
    def get_field_count(self, category: Optional[str] = None) -> int:
        """
        Get count of fields in a category or total across all categories.
        
        Args:
            category: Specific category to count, or None for total count
            
        Returns:
            Number of fields
        """
        if category is not None:
            if category not in self.categories:
                return 0
            return len(self.categories[category])
        
        # Total across all categories
        return sum(len(fields) for fields in self.categories.values())
    
    def __str__(self) -> str:
        """String representation of FieldManager."""
        total_fields = self.get_field_count()
        categories_summary = ", ".join(
            f"{cat}({len(fields)})" for cat, fields in self.categories.items()
        )
        return f"FieldManager({len(self.categories)} categories, {total_fields} total fields: {categories_summary})"


def load_fields(csv_path: str, category: Optional[str] = None) -> Dict[str, Dict]:
    """Load field specs from CSV, suitable for passing to ``LLMStep(fields=...)``.

    Args:
        csv_path: Path to the field categories CSV file.
        category: If provided, return only fields in this category.
            If None, return all fields merged across categories.

    Returns:
        Dict mapping field names to their specs (prompt, type, instructions).

    Example::

        from lattice.data import load_fields
        fields = load_fields("fields.csv", category="business_analysis")
        pipeline = Pipeline([LLMStep("analyze", fields=fields)])
    """
    fm = FieldManager(csv_path)
    if category:
        return fm.get_category_fields(category)
    all_fields: Dict[str, Dict] = {}
    for cat in fm.get_categories():
        all_fields.update(fm.get_category_fields(cat))
    return all_fields