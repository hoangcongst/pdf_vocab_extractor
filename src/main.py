#!/usr/bin/env python3
"""
Korean Language Extraction and Analysis Tool

This tool extracts Korean vocabulary and grammar from PDF files, processes them
using GPT-4o-mini, and exports the results to an Excel file for learners at
TOPIK 3-4 (intermediate) level.
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
from .text_processor.korean_processor import process_korean_text
from .gpt_integration.openai_client import process_with_openai, format_results_to_text
from .gpt_integration.openai_batch_processor import process_and_deduplicate
from .export.excel_exporter import export_to_csv


# Load environment variables
load_dotenv()


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract Korean vocabulary and grammar from PDF files.'
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
        '-b', 
        '--batch-size', 
        type=int, 
        default=10,
        help='Batch size for GPT processing (default: 10)'
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
        '--mecab',
        action='store_true',
        help='Use Mecab tokenizer instead of Okt (default: False)'
    )
    parser.add_argument(
        '--skip-gpt',
        action='store_true',
        help='Skip GPT processing (useful for testing)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of vocabulary items to process (for testing)'
    )
    parser.add_argument(
        '--dedupe-batch',
        action='store_true',
        help='Deduplicate and normalize words using GPT-4o-mini batch processing'
    )
    parser.add_argument(
        '--dedupe-batch-size',
        type=int,
        default=200,
        help='Batch size for deduplication processing (default: 200)'
    )
    
    return parser.parse_args()


def process_data(args) -> Dict[str, Any]:
    """
    Process the input data and return results.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary containing processed results
    """
    # Extract text from PDF
    pages_text = extract_text_from_pdf(args.input)
    
    # Process text to extract vocabulary and grammar
    all_vocabulary = set()
    all_grammar = []
    
    # Process each page
    for page_text in pages_text:
        result = process_korean_text(page_text)
        all_vocabulary.update(result['vocabulary'])
        all_grammar.extend(result['grammar'])
    
    # Convert to expected format
    extracted_data = {
        'vocabulary': sorted(list(all_vocabulary)),
        'grammar': all_grammar
    }
    
    # Skip GPT processing if requested
    if args.skip_gpt:
        return {
            'vocabulary_results': [{'item': word, 'analysis': '', 'model': 'none'} for word in extracted_data['vocabulary']],
            'grammar_results': [{'item': pattern, 'analysis': '', 'model': 'none'} for pattern in extracted_data['grammar']]
        }
    
    # Process with OpenAI
    return process_with_openai(extracted_data, batch_size=args.batch_size)


def main():
    """Main entry point for the application."""
    start_time = time.time()
    args = parse_arguments()
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PDF file: {args.input}")
    
    # Step 1: Extract text from PDF
    logger.info(f"Extracting text using {args.method}...")
    pages_text = extract_text_from_pdf(args.input, prefer_method=args.method)
    logger.info(f"Extracted {len(pages_text)} pages")
    
    # Step 2: Process Korean text to identify vocabulary and grammar
    logger.info("Processing Korean text to extract vocabulary and grammar...")
    
    all_vocabulary = set()
    all_grammar = []
    
    for i, page_text in enumerate(pages_text):
        logger.info(f"Processing page {i+1}/{len(pages_text)}")
        result = process_korean_text(page_text, use_mecab=args.mecab)
        
        # Add unique vocabulary
        all_vocabulary.update(result['vocabulary'])
        
        # Add grammar examples
        all_grammar.extend(result['grammar'])
    
    # Convert vocabulary set to sorted list
    vocabulary_list = sorted(list(all_vocabulary))
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        logger.info(f"Limiting vocabulary to {args.limit} items")
        vocabulary_list = vocabulary_list[:args.limit]
    
    logger.info(f"Total unique vocabulary items: {len(vocabulary_list)}")
    logger.info(f"Total grammar examples: {len(all_grammar)}")
    
    # NEW STEP: Process words with batch deduplication if requested
    if args.dedupe_batch:
        logger.info(f"Performing batch deduplication and normalization with batch size {args.dedupe_batch_size}...")
        deduplicated_vocabulary = process_and_deduplicate(vocabulary_list, batch_size=args.dedupe_batch_size)
        logger.info(f"Vocabulary size reduced from {len(vocabulary_list)} to {len(deduplicated_vocabulary)} items")
        vocabulary_list = deduplicated_vocabulary
    
    # Step 3: Process items with GPT-4o-mini
    if not args.skip_gpt:
        logger.info("Processing with GPT-4o-mini...")
        
        # Process vocab and grammar with OpenAI
        processed_data = process_data(args)
        
        # Step 4: Export results to CSV
        logger.info(f"Exporting results to CSV: {args.output}")
        output_paths = export_to_csv(processed_data, args.output)
        logger.info(f"Results saved to: {output_paths}")
    else:
        logger.info("Skipped GPT processing as requested")
        # Create simple data structure for CSV export
        processed_data = {
            'vocabulary_results': [
                {'item': word, 'analysis': '', 'model': 'N/A', 'error': False}
                for word in vocabulary_list
            ],
            'grammar_results': [
                {'item': f"문법: {pattern}\n예문: {example}", 'analysis': '', 'model': 'N/A', 'error': False}
                for pattern, example in all_grammar
            ]
        }
        # Export to CSV
        logger.info(f"Exporting results to CSV: {args.output}")
        output_paths = export_to_csv(processed_data, args.output)
        logger.info(f"Results saved to: {output_paths}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
