"""
Korean Text Processor Module

This module handles extraction and processing of Korean vocabulary and grammar from text.
"""

import re
import logging
from typing import List, Dict, Set, Tuple
import os

from pathlib import Path


logger = logging.getLogger(__name__)


# Try to import KoNLPy, but provide fallbacks if not available
try:
    from konlpy.tag import Mecab, Okt
    KONLPY_AVAILABLE = True
except ImportError:
    logger.warning("KoNLPy not available. Using regex-based fallback methods.")
    KONLPY_AVAILABLE = False


class KoreanTextProcessor:
    """
    Class for processing Korean text to extract vocabulary and grammar.
    Uses KoNLPy for tokenization and lemmatization.
    """
    
    # Common Korean particles and endings to help identify word boundaries
    PARTICLES = {
        '은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로',
        '와', '과', '이나', '나', '이랑', '랑', '만', '까지', '부터', '도'
    }
    
    # TOPIK 3-4 level grammar patterns (example, should be extended)
    GRAMMAR_PATTERNS = [
        r'(은|는|을|를) 것 같다',
        r'(으)?ㄹ 수 있다',
        r'(으)?ㄹ 것이다',
        r'(아|어|여) 보다',
        r'(아|어|여) 주다',
        r'기 때문에',
        r'(으)?려고 하다',
        r'(으)?면 안 되다',
        r'(으)?면서',
        r'(아|어|여)도 되다',
        r'(아|어|여)야 하다',
        r'(으)?ㄴ 적이 있다',
        r'(으)?ㄹ 때',
        r'(으)?니까',
        r'(으)?ㄴ/는데',
        r'지만',
        r'아/어/여서',
        r'(으)?ㄹ까요',
        r'(으)?ㄹ래요',
        r'(는)군요',
        r'거든요',
        r'(으)ㄹ게요',
        r'(아|어|여)야겠다',
        r'(아|어|여)도',
        r'(으)?ㄹ까 하다',
        r'(으)?면 좋겠다',
        r'(으)?려면',
        r'(으)?ㄴ/는 것 같다',
        r'(으)?ㄴ 덕분에',
        r'(으)?ㄹ 텐데'
    ]
    
    def __init__(self, use_mecab=False):
        """
        Initialize the Korean text processor.
        
        Args:
            use_mecab: Whether to use Mecab (True) or Okt (False) tokenizer
                       (only used if KoNLPy is available)
        """
        self.konlpy_available = KONLPY_AVAILABLE
        
        if self.konlpy_available:
            try:
                if use_mecab:
                    self.tokenizer = Mecab()
                    logger.info("Using Mecab tokenizer")
                else:
                    self.tokenizer = Okt()
                    logger.info("Using Okt tokenizer")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer: {e}")
                self.konlpy_available = False
                logger.info("Falling back to regex-based methods")
        
        # Compile grammar patterns for faster matching
        self.compiled_patterns = [re.compile(pattern) for pattern in self.GRAMMAR_PATTERNS]
        
        # Pattern to extract Korean words
        self.word_pattern = re.compile(r'[가-힣]+')
        
    def clean_text(self, text: str) -> str:
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
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract individual sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting by punctuation
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() + '.' for s in sentences if s.strip()]
    
    def normalize_korean_word(self, word: str) -> str:
        """
        Normalize a Korean word by removing conjugations and particles.
        
        Args:
            word: Korean word to normalize
            
        Returns:
            Normalized word
        """
        # If using Okt, try to get the base form using pos tagging
        if self.konlpy_available and isinstance(self.tokenizer, Okt):
            try:
                # Try to get the normalized form
                norm = self.tokenizer.normalize(word)
                # Get the first lemma using morphs
                if norm != word:
                    return norm
                
                # Use pos tagging to find lemma
                pos_result = self.tokenizer.pos(word, norm=True)
                if pos_result:
                    # Extract the base form
                    lemma = pos_result[0][0]
                    return lemma
            except:
                pass  # Fall back to regex if Okt fails
                
        # Regex fallback for simple normalization
        # Remove common Korean particles
        for particle in sorted(self.PARTICLES, key=len, reverse=True):
            if word.endswith(particle) and len(word) > len(particle):
                return word[:-len(particle)]
                
        return word
            
    def extract_vocabulary_konlpy(self, text: str) -> List[str]:
        """
        Extract unique Korean vocabulary words using KoNLPy.
        
        Args:
            text: Input text
            
        Returns:
            List of unique vocabulary words
        """
        # Clean the text first
        clean = self.clean_text(text)
        
        # Extract words based on the tokenizer type
        all_words = set()
        
        if isinstance(self.tokenizer, Okt):
            # For Okt: extract nouns, verbs, adjectives
            nouns = self.tokenizer.nouns(clean)
            all_words.update(nouns)
            
            # Extract verbs and adjectives as lemmas
            pos_tagged = self.tokenizer.pos(clean, norm=True)
            verbs_adjs = [word for word, pos in pos_tagged 
                         if pos.startswith('V') or pos.startswith('J')]
            all_words.update(verbs_adjs)
            
        elif isinstance(self.tokenizer, Mecab):
            # For Mecab: use pos tagging to extract content words
            pos_tagged = self.tokenizer.pos(clean)
            content_words = [word for word, pos in pos_tagged 
                           if pos.startswith('N') or pos.startswith('V') or pos.startswith('M')]
            all_words.update(content_words)
            
        # Remove duplicates and normalize
        vocabulary = []
        normalized_words = set()
        
        for word in all_words:
            if len(word) > 1:  # Skip single characters
                norm_word = self.normalize_korean_word(word)
                if norm_word and norm_word not in normalized_words:
                    normalized_words.add(norm_word)
                    vocabulary.append(word)  # Keep original for display
                    
        return vocabulary
    
    def extract_vocabulary_regex(self, text: str) -> List[str]:
        """
        Extract unique Korean vocabulary words using regex (fallback method).
        
        Args:
            text: Input text
            
        Returns:
            List of unique vocabulary words
        """
        # Clean the text first
        clean = self.clean_text(text)
        
        # Split by whitespace to get rough words
        words = clean.split()
        
        # Extract Korean characters
        korean_words = set()
        for word in words:
            # Extract only Korean part (remove punctuation)
            matches = self.word_pattern.findall(word)
            for match in matches:
                if len(match) > 1:  # Skip single characters
                    korean_words.add(match)
        
        return list(korean_words)
    
    def extract_vocabulary(self, text: str) -> List[str]:
        """
        Extract unique Korean vocabulary words from text.
        
        Args:
            text: Input text
            
        Returns:
            List of unique vocabulary words
        """
        if self.konlpy_available:
            vocabulary = self.extract_vocabulary_konlpy(text)
        else:
            vocabulary = self.extract_vocabulary_regex(text)
        
        # Filter very short words (likely particles or single characters)
        vocabulary = [word for word in vocabulary if len(word) > 1]
        
        return sorted(vocabulary)
    
    def extract_grammar_patterns(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract grammar patterns and example sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of (grammar_pattern, example_sentence) tuples
        """
        sentences = self.extract_sentences(text)
        grammar_examples = []
        
        for sentence in sentences:
            # Check each grammar pattern against this sentence
            for pattern_index, pattern in enumerate(self.compiled_patterns):
                if pattern.search(sentence):
                    # Found a match
                    grammar_pattern = self.GRAMMAR_PATTERNS[pattern_index]
                    grammar_examples.append((grammar_pattern, sentence))
                    break  # Move to next sentence after finding a match
        
        return grammar_examples
    
    def process_text(self, text: str) -> Dict:
        """
        Process Korean text to extract vocabulary and grammar.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with vocabulary and grammar
        """
        vocabulary = self.extract_vocabulary(text)
        grammar = self.extract_grammar_patterns(text)
        
        return {
            'vocabulary': vocabulary,
            'grammar': grammar
        }


def process_korean_text(text: str, use_mecab: bool = False) -> Dict:
    """
    Convenience function to process Korean text.
    
    Args:
        text: Input text
        use_mecab: Whether to use Mecab tokenizer (if available)
        
    Returns:
        Dictionary with vocabulary and grammar
    """
    processor = KoreanTextProcessor(use_mecab=use_mecab)
    return processor.process_text(text) 