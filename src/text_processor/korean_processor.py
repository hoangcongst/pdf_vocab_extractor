"""
Korean Text Processor Module

This module handles extraction and processing of Korean vocabulary using KiwiPiepy.
"""

import re
import logging
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Import KiwiPiepy
try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
    kiwi = Kiwi()
    logger.info("KiwiPiepy loaded successfully")
except ImportError:
    logger.warning("KiwiPiepy not available")
    KIWI_AVAILABLE = False

def clean_text(text: str) -> str:
    """
    Clean and normalize text before processing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove non-Korean characters except spaces and some punctuation
    text = re.sub(r'[^\s가-힣.,?!:;()"]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def parse_korean_text(text: str) -> Dict:
    """
    Parse Korean text using KiwiPiepy, focusing on main word types.
    Adds '다' after adjectives and verbs to make them dictionary form.
    
    Args:
        text: Korean text to parse
        
    Returns:
        Dictionary containing categorized words
    """
    if not KIWI_AVAILABLE:
        logger.warning("KiwiPiepy not available for parsing")
        return {
            'nouns': [],
            'verbs': [],
            'adjectives': [],
            'adverbs': []
        }
    
    try:
        # Clean the text first
        clean = clean_text(text)
        
        # Tokenize the text
        tokens = kiwi.tokenize(clean)
        
        # Initialize categories
        result = {
            'nouns': set(),      # NNG, NNP
            'verbs': set(),      # VV, VX
            'adjectives': set(), # VA
            'adverbs': set()     # MAG
        }
        
        for token in tokens:
            pos = token.tag
            word = token.form
            
            # Skip single character words
            if len(word) < 2:
                continue
                
            # Categorize words and add '다' where appropriate
            if pos.startswith('NN'):  # Nouns (NNG: common noun, NNP: proper noun)
                result['nouns'].add(word)
            elif pos.startswith('VV') or pos.startswith('VX'):  # Verbs
                result['verbs'].add(f"{word}다")
            elif pos.startswith('VA'):  # Adjectives
                result['adjectives'].add(f"{word}다")
            elif pos.startswith('MAG'):  # Adverbs
                result['adverbs'].add(word)
        
        # Convert sets to sorted lists
        return {
            category: sorted(words)
            for category, words in result.items()
        }
        
    except Exception as e:
        logger.error(f"Error parsing with KiwiPiepy: {str(e)}")
        return {
            'nouns': [],
            'verbs': [],
            'adjectives': [],
            'adverbs': []
        } 