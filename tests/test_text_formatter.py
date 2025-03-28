"""
Test Text Formatter Module

This module tests the text formatting functionality for Korean vocabulary and grammar.
"""

import sys
from pathlib import Path
import json

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.gpt_integration.openai_client import format_word_analysis, format_results_to_text
from tests.sample_data import SAMPLE_VOCAB_RESULTS, SAMPLE_GRAMMAR_RESULTS


def test_format_word_analysis():
    """Test formatting a single word's analysis."""
    # Create a sample word data
    word_data = {
        "word": "한국어",
        "meanings": ["Tiếng Hàn Quốc"],
        "examples": {
            "meaning1": [
                {
                    "korean": "저는 한국어를 공부해요",
                    "vietnamese": "Tôi học tiếng Hàn"
                }
            ]
        },
        "memory_tip": "Liên tưởng 한국 là tên nước và 어 là ngôn ngữ, kết hợp lại là \"tiếng của nước Hàn\"",
        "hanja_analysis": {
            "explanation": "한국 (Hàn Quốc) + 어 (ngữ/tiếng)",
            "related_words": ["중국어", "일본어"]
        }
    }
    
    formatted_text = format_word_analysis(word_data)
    print("\nSingle Word Analysis Example:")
    print(formatted_text)
    
    assert "Từ: 한국어" in formatted_text
    assert "Nghĩa:" in formatted_text
    assert "Ví dụ:" in formatted_text
    assert "Tip để nhớ từ:" in formatted_text
    assert "Phân tích Hán tự:" in formatted_text


def test_format_results_to_text():
    """Test formatting multiple word analysis results."""
    # Use sample data directly
    formatted_text = format_results_to_text(SAMPLE_VOCAB_RESULTS[:3])
    print("\nMultiple Words Analysis Example:")
    print(formatted_text)
    
    # Basic assertions
    assert formatted_text  # Should not be empty
    assert "한국어" in formatted_text  # First word should be present
    assert "공부하다" in formatted_text  # Second word should be present
    assert "========================================" in formatted_text  # Separator should be present


def main():
    """Run the formatting examples."""
    print("Testing Text Formatting Functions")
    print("=" * 50)
    
    # Test single word formatting
    test_format_word_analysis()
    
    # Test multiple words formatting
    test_format_results_to_text()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main() 