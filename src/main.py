#!/usr/bin/env python3
"""
Korean Language Extraction and Analysis Tool

This tool extracts Korean vocabulary from PDF files, processes them using KiwiPiepy,
and exports the results to CSV files with ChatGPT explanations.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import time
from dotenv import load_dotenv
from typing import Dict, Any

from .pdf_extractor.pdf_reader import extract_text_from_pdf
from .text_processor.korean_processor import parse_korean_text
from .gpt_integration.openai_client import process_with_openai
from .export.excel_exporter import export_to_csv

# Load environment variables
load_dotenv()

def setup_logging(level='INFO'):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract Korean vocabulary from PDF files using KiwiPiepy.'
    )
    parser.add_argument(
        'input', 
        type=str, 
        help='Path to the PDF file to process'
    )
    parser.add_argument(
        '-o', 
        '--output', 
        type=str, 
        default='output.csv',
        help='Output CSV file path (default: output.csv)'
    )
    parser.add_argument(
        '-m',
        '--method',
        type=str,
        choices=['pdfplumber', 'pypdf2'],
        default='pdfplumber',
        help='PDF extraction method (default: pdfplumber)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of vocabulary items to process (for testing)'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for GPT processing (default: 10)'
    )
    parser.add_argument(
        '--skip-gpt',
        action='store_true',
        help='Skip GPT processing (useful for testing)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    return parser.parse_args()

def save_text_to_file(text: str, output_path: str):
    """
    Save extracted text to a file.
    
    Args:
        text: Text to save
        output_path: Path to save the text file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Text saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving text to file: {str(e)}")

def process_data(args) -> Dict[str, Any]:
    """
    Process the input data and return results.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary containing processed results
    """
    logger = logging.getLogger(__name__)
    
    # Extract text from PDF
    pages_text = extract_text_from_pdf(args.input)
    
    # Save combined text to file
    combined_text = "\n\n=== PAGE BREAK ===\n\n".join(pages_text)
    txt_output = args.output.replace('.csv', '.txt')
    save_text_to_file(combined_text, txt_output)
    
    # Process text to extract vocabulary
    all_words = {
        'nouns': set(),
        'verbs': set(),
        'adjectives': set(),
        'adverbs': set()
    }
    
    # Process each page
    for page_text in pages_text:
        result = parse_korean_text(page_text)
        for category in all_words:
            all_words[category].update(result[category])
    
    # Convert sets to sorted lists
    vocabulary_list = []
    for category, words in all_words.items():
        for word in sorted(words):
            vocabulary_list.append(word)
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        vocabulary_list = vocabulary_list[:args.limit]
    
    # Skip GPT processing if requested
    if args.skip_gpt:
        return {
            'vocabulary_results': [
                {'item': word, 'analysis': '', 'model': 'none'} 
                for word in vocabulary_list
            ]
        }
    
    # Process with OpenAI
    logger.info("Processing vocabulary with GPT...")
    processed_data = process_with_openai(
        {'vocabulary': vocabulary_list},
        batch_size=args.batch_size
    )
    
    # Debug log
    import json
    logger.debug("OpenAI Response:")
    logger.debug(json.dumps(processed_data, indent=2, ensure_ascii=False))
    
    # Add category information to results
    for result in processed_data['vocabulary_results']:
        word = result['item']
        for category, words in all_words.items():
            if word in words:
                result['category'] = category
                break
    
    # Format and save HTML output
    from .gpt_integration.openai_client import format_results_to_text
    html_output = format_results_to_text(processed_data['vocabulary_results'])
    html_path = args.output.replace('.csv', '.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Korean Vocabulary Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        .word-analysis { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }
        .word { color: #2c3e50; margin-top: 0; }
        .meanings h4, .examples h4, .memory-tip h4, .hanja-analysis h4, .grammar-points h4 { color: #3498db; }
        .korean { color: #e74c3c; }
        .vietnamese { color: #27ae60; }
        ul { padding-left: 20px; }
        li { margin: 5px 0; }
    </style>
</head>
<body>
""")
        f.write(html_output)
        f.write("\n</body>\n</html>")
    logger.info(f"HTML output saved to: {html_path}")
    
    return processed_data

def main():
    """Main entry point for the application."""
    start_time = time.time()
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PDF file: {args.input}")
    
    # Step 1: Extract text from PDF and save to txt
    logger.info(f"Extracting text using {args.method}...")
    processed_data = process_data(args)
    
    # Get counts by category
    category_counts = {}
    for result in processed_data['vocabulary_results']:
        category = result['category']
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Log statistics
    logger.info("Extraction Statistics:")
    for category, count in category_counts.items():
        logger.info(f"Total {category}: {count}")
    
    # Export results to CSV
    logger.info(f"Exporting results to CSV: {args.output}")
    output_paths = export_to_csv(processed_data, args.output)
    logger.info(f"Results saved to: {output_paths}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    return 0

if __name__ == "__main__":
    sys.exit(main())
