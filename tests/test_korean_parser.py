"""
Test Korean Parser Module

This module tests Korean text parsing using KiwiPiepy
"""

import sys
from pathlib import Path
import logging
from typing import Dict

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def parse_korean_text(text: str) -> Dict:
    """
    Parse Korean text using KiwiPiepy, focusing on main word types.
    
    Args:
        text: Korean text to parse
        
    Returns:
        Dictionary containing categorized words
    """
    if not KIWI_AVAILABLE:
        logger.warning("KiwiPiepy not available for parsing")
        return {}
    
    try:
        # Tokenize the text
        tokens = kiwi.tokenize(text)
        
        # Initialize categories
        result = {
            'nouns': [],      # NNG, NNP
            'verbs': [],      # VV, VX
            'adjectives': [], # VA
            'adverbs': [],    # MAG
        }
        
        for token in tokens:
            pos = token.tag
            word = token.form
            
            # Categorize words
            if pos.startswith('NN'):  # Nouns (NNG: common noun, NNP: proper noun)
                result['nouns'].append(word)
            elif pos.startswith('VV') or pos.startswith('VX'):  # Verbs
                result['verbs'].append(word)
            elif pos.startswith('VA'):  # Adjectives
                result['adjectives'].append(word)
            elif pos.startswith('MAG'):  # Adverbs
                result['adverbs'].append(word)
        
        # Remove duplicates and sort
        for category in result:
            result[category] = sorted(list(set(result[category])))
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing with KiwiPiepy: {str(e)}")
        return {}

def test_korean_parsing():
    """Test KiwiPiepy parsing with diverse Korean sentences."""
    # Test sentences covering various patterns
    sentences = [
        # Basic patterns
        "저는 한국어를 열심히 공부합니다.",  # Simple sentence with adverb
        "어제 친구와 재미있게 놀았어요.",  # Past tense with adjective
        "내일 시장에서 맛있는 음식을 먹을 거예요.",  # Future with descriptive
        
        # Compound sentences
        "비가 많이 오지만 학교에 꼭 가야 해요.",  # Contrast with adverbs
        "한국어가 어렵지만 재미있게 배워요.",  # Descriptive with learning
    ]
    
    print("\nKorean Text Analysis")
    print("=" * 50)
    
    for sentence in sentences:
        print(f"\n문장 (Sentence):")
        print(f"{sentence}")
        
        # Get parsing results
        result = parse_korean_text(sentence)
        
        # Print categorized results
        print("\n품사별 분석 (Part of Speech Analysis):")
        
        if result['nouns']:
            print("\n명사 (Nouns):")
            print(", ".join(result['nouns']))
            
        if result['verbs']:
            print("\n동사 (Verbs):")
            print(", ".join(result['verbs']))
            
        if result['adjectives']:
            print("\n형용사 (Adjectives):")
            print(", ".join(result['adjectives']))
            
        if result['adverbs']:
            print("\n부사 (Adverbs):")
            print(", ".join(result['adverbs']))
        
        print("-" * 50)

def main():
    """Run the Korean parser test."""
    print("Korean Parser Analysis")
    print("=" * 50)
    
    # Check if KiwiPiepy is available
    if not KIWI_AVAILABLE:
        print("\nError: KiwiPiepy is not available")
        return
    
    # Run the focused parsing test
    test_korean_parsing()
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main() 