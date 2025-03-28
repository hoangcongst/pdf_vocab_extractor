"""
CSV Export Module

This module handles exporting processed Korean vocabulary to CSV files.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

class CSVExporter:
    """Class to export data to CSV files."""
    
    def __init__(self, output_path=None):
        """
        Initialize the CSV exporter.
        
        Args:
            output_path: Base path for output CSV files
        """
        if output_path:
            self.output_dir = Path(output_path).parent
            self.base_name = Path(output_path).stem
        else:
            self.output_dir = Path('.')
            self.base_name = "korean_vocabulary"

    def format_vocabulary_data(self, vocabulary_results: List[Dict]) -> pd.DataFrame:
        """
        Format vocabulary results for CSV export.
        
        Args:
            vocabulary_results: List of dictionaries with vocabulary data
            
        Returns:
            Pandas DataFrame with formatted data
        """
        data = []
        
        for item in vocabulary_results:
            # Format HTML analysis
            from ..gpt_integration.openai_client import format_word_analysis
            html_analysis = format_word_analysis(item)
            
            data.append({
                'Word': item['item'],
                'Category': item['category'],
                'Analysis': item.get('analysis', ''),
                'HTML_Analysis': html_analysis
            })
        
        return pd.DataFrame(data)
    
    def export(self, data: Dict) -> Dict[str, str]:
        """
        Export data to CSV files.
        
        Args:
            data: Dictionary containing vocabulary results
            
        Returns:
            Dictionary with paths to the exported CSV files
        """
        output_paths = {}
        
        # Format and export vocabulary
        if 'vocabulary_results' in data:
            vocab_df = self.format_vocabulary_data(data['vocabulary_results'])
            vocab_path = self.output_dir / f"{self.base_name}_vocabulary.csv"
            vocab_df.to_csv(vocab_path, index=False, encoding='utf-8', quoting=1, header=False)
            output_paths['vocabulary'] = str(vocab_path)
        
        return output_paths

def export_to_csv(data: Dict, output_path: str) -> Dict[str, str]:
    """
    Convenience function to export data to CSV files.
    
    Args:
        data: Dictionary containing vocabulary results
        output_path: Base path for the CSV files
        
    Returns:
        Dictionary with paths to the exported CSV files
    """
    exporter = CSVExporter(output_path)
    return exporter.export(data) 