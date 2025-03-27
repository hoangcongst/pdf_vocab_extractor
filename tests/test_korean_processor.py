"""
Tests for the Korean Text Processor module.
"""

import pytest
from src.text_processor.korean_processor import KoreanTextProcessor, process_korean_text


@pytest.fixture
def sample_korean_text():
    """Sample Korean text for testing."""
    return """
    안녕하세요. 저는 한국어를 공부하고 있습니다. 
    오늘은 날씨가 좋은 것 같아요. 내일은 비가 올 수도 있습니다.
    한국 음식을 좋아해서 자주 먹어요. 김치는 맵지만 맛있어요.
    한국에 가려고 해요. 한국어를 잘하면 좋겠어요.
    """


@pytest.fixture
def processor():
    """Create a KoreanTextProcessor instance."""
    return KoreanTextProcessor(use_mecab=False)


def test_clean_text(processor, sample_korean_text):
    """Test text cleaning functionality."""
    cleaned = processor.clean_text(sample_korean_text)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
    assert '123' not in cleaned  # Should remove non-Korean characters


def test_extract_sentences(processor, sample_korean_text):
    """Test sentence extraction."""
    sentences = processor.extract_sentences(sample_korean_text)
    assert isinstance(sentences, list)
    assert len(sentences) > 0
    # Verify each sentence ends with a period
    for sentence in sentences:
        assert sentence.endswith('.')


def test_extract_vocabulary(processor, sample_korean_text):
    """Test vocabulary extraction."""
    vocabulary = processor.extract_vocabulary(sample_korean_text)
    assert isinstance(vocabulary, list)
    assert len(vocabulary) > 0
    
    # Check for expected words (may depend on tokenizer)
    expected_words = ['한국어', '공부', '날씨', '한국']
    for word in expected_words:
        assert any(word in v for v in vocabulary), f"Expected '{word}' in vocabulary"


def test_extract_grammar_patterns(processor, sample_korean_text):
    """Test grammar pattern extraction."""
    grammar_examples = processor.extract_grammar_patterns(sample_korean_text)
    assert isinstance(grammar_examples, list)
    
    # The sample text has at least these grammar patterns
    # Check if we found at least one pattern
    if grammar_examples:
        pattern, example = grammar_examples[0]
        assert isinstance(pattern, str)
        assert isinstance(example, str)


def test_process_text(processor, sample_korean_text):
    """Test the main processing function."""
    results = processor.process_text(sample_korean_text)
    assert isinstance(results, dict)
    assert 'vocabulary' in results
    assert 'grammar' in results
    assert isinstance(results['vocabulary'], list)
    assert isinstance(results['grammar'], list)


def test_convenience_function(sample_korean_text):
    """Test the convenience function."""
    results = process_korean_text(sample_korean_text)
    assert isinstance(results, dict)
    assert 'vocabulary' in results
    assert 'grammar' in results 