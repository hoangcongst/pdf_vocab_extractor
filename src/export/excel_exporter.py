"""
CSV Export Module

This module handles exporting processed Korean vocabulary and grammar to CSV files.
"""

import os
import logging
import json
from typing import List, Dict, Any
from pathlib import Path
import datetime

import pandas as pd

from ..gpt_integration.openai_client import format_word_analysis

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
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path('.')
            self.base_name = f"korean_vocab_grammar_{timestamp}"

    def format_analysis_as_html(self, text: str) -> str:
        """
        Format analysis text as HTML using basic tags.
        
        Args:
            text: Raw analysis text
            
        Returns:
            HTML formatted text
        """
        if not text:
            return ""
        
        # Split into sections
        sections = text.split('\n\n')
        html_parts = ['<div>']
        
        for section in sections:
            if not section.strip():
                continue
                
            # Handle section titles
            if section.startswith('Từ:'):
                html_parts.append('<h3>Từ vựng</h3>')
                continue
            elif section.startswith('Nghĩa:'):
                html_parts.append('<h4>Nghĩa</h4>')
                lines = section.split('\n')[1:]  # Skip the "Nghĩa:" line
                html_parts.append('<ul>')
                for line in lines:
                    if line.strip():
                        html_parts.append(f'<li>{line.strip()}</li>')
                html_parts.append('</ul>')
            elif section.startswith('Ví dụ:'):
                html_parts.append('<h4>Ví dụ</h4>')
                lines = section.split('\n')[1:]  # Skip the "Ví dụ:" line
                html_parts.append('<ul>')
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        korean = lines[i].strip().lstrip('- ')
                        vietnamese = lines[i + 1].strip()
                        html_parts.append(f'<li><b>{korean}</b><br>{vietnamese}</li>')
                html_parts.append('</ul>')
            elif section.startswith('Tip để nhớ từ:'):
                html_parts.append('<h4>Tip để nhớ từ</h4>')
                content = section.split('\n', 1)[1].strip()
                html_parts.append(f'<p>{content}</p>')
            elif section.startswith('Phân tích Hán tự:'):
                html_parts.append('<h4>Phân tích Hán tự</h4>')
                content = section.split('\n', 1)[1].strip()
                html_parts.append(f'<p>{content}</p>')
            elif section.startswith('Ngữ pháp:'):
                html_parts.append('<h4>Ngữ pháp</h4>')
                lines = section.split('\n')[1:]
                html_parts.append('<ul>')
                for line in lines:
                    if line.strip():
                        html_parts.append(f'<li>{line.strip()}</li>')
                html_parts.append('</ul>')
            else:
                # Generic section
                html_parts.append(f'<p>{section.strip()}</p>')
        
        html_parts.append('</div>')
        return ''.join(html_parts)
    
    def format_vocabulary_data(self, vocabulary_results: List[Dict]) -> pd.DataFrame:
        """
        Format vocabulary results for CSV export.
        
        Args:
            vocabulary_results: List of dictionaries with vocabulary analysis
            
        Returns:
            Pandas DataFrame with formatted data
        """
        data = []
        
        for item in vocabulary_results:
            if isinstance(item.get('analysis'), dict):
                # Format the analysis into readable text
                formatted_analysis = format_word_analysis(item['analysis'])
            else:
                formatted_analysis = str(item.get('analysis', ''))
            
            # Format analysis as HTML
            html_analysis = self.format_analysis_as_html(formatted_analysis)
            
            data.append({
                'Word': item['item'],
                'Analysis': html_analysis
            })
        
        return pd.DataFrame(data)
    
    def format_grammar_data(self, grammar_results: List[Dict]) -> pd.DataFrame:
        """
        Format grammar results for CSV export.
        
        Args:
            grammar_results: List of dictionaries with grammar analysis
            
        Returns:
            Pandas DataFrame with formatted data
        """
        data = []
        
        for item in grammar_results:
            if isinstance(item.get('analysis'), dict):
                # Format the analysis into readable text
                formatted_analysis = format_word_analysis(item['analysis'])
            else:
                formatted_analysis = str(item.get('analysis', ''))
            
            # Format analysis as HTML
            html_analysis = self.format_analysis_as_html(formatted_analysis)
            
            data.append({
                'Pattern': item['item'],
                'Analysis': html_analysis
            })
        
        return pd.DataFrame(data)
    
    def export(self, data: Dict) -> Dict[str, str]:
        """
        Export data to CSV files.
        
        Args:
            data: Dictionary containing vocabulary and grammar results
            
        Returns:
            Dictionary with paths to the exported CSV files
        """
        output_paths = {}
        
        # Format and export vocabulary
        if 'vocabulary_results' in data:
            vocab_df = self.format_vocabulary_data(data['vocabulary_results'])
            vocab_path = self.output_dir / f"{self.base_name}_vocabulary.csv"
            vocab_df.to_csv(vocab_path, index=False, encoding='utf-8', quoting=1, header=False)  # Skip header
            output_paths['vocabulary'] = str(vocab_path)
        
        # Format and export grammar
        if 'grammar_results' in data:
            grammar_df = self.format_grammar_data(data['grammar_results'])
            grammar_path = self.output_dir / f"{self.base_name}_grammar.csv"
            grammar_df.to_csv(grammar_path, index=False, encoding='utf-8', quoting=1, header=False)  # Skip header
            output_paths['grammar'] = str(grammar_path)
        
        return output_paths


def export_to_csv(data: Dict, output_path: str) -> Dict[str, str]:
    """
    Convenience function to export data to CSV files.
    
    Args:
        data: Dictionary containing vocabulary and grammar results
        output_path: Base path for the CSV files
        
    Returns:
        Dictionary with paths to the exported CSV files
    """
    exporter = CSVExporter(output_path)
    return exporter.export(data) 