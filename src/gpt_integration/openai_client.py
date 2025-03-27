"""
OpenAI Integration Module

This module handles integration with OpenAI API for processing Korean vocabulary and grammar.
"""

import os
import logging
import time
from typing import List, Dict, Any
from pathlib import Path
import json

import openai
from tqdm import tqdm
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Load API key from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Check if API key is available
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables.")


class OpenAIProcessor:
    """Class to process text using OpenAI API."""
    
    # The standard prompt for Korean-Vietnamese translation and analysis
    DEFAULT_PROMPT = """
    Bạn là từ điển AI dịch từ tiếng Hàn sang tiếng Việt, tôi gửi bạn 1 từ tiếng Hàn, bạn sẽ đưa ra các nghĩa cho tôi và kèm ví dụ, tip để tôi có thể nhớ được từ.                           
    - Nếu là từ Hán Hàn, hãy phân tích từ tiếng Hán cấu tạo nên từ đó. Ví dụ: 방법. 방 là phương, 법 là pháp nên 방법 là phương pháp.
    Nêu 1 số ví dụ từ Hán Hàn được cấu tạo từ từ cấu thành nên từ gốc.                             
    - Nếu input là cấu trúc ngữ pháp hoặc câu đầy đủ thì Phân tích các cấu trúc ngữ pháp nếu có xuất hiện dành cho trình độ topik trung cấp 1, 2
    - Không gửi nội dung thừa, không phiên âm
    """
    
    def __init__(self, api_key=None, model=None, prompt=None):
        """
        Initialize the OpenAI processor.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model to use (defaults to environment variable or gpt-4o-mini)
            prompt: System prompt template (defaults to Korean-Vietnamese translation)
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            raise ValueError("OpenAI API key is required")
        
        self.model = model or OPENAI_MODEL
        logger.info(f"Using OpenAI model: {self.model}")
        
        self.system_prompt = prompt or self.DEFAULT_PROMPT
        
        # Initialize OpenAI client - using only new API style (v1.0.0+)
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def process_item(self, item: str) -> Dict:
        """
        Process a single vocabulary or grammar item.
        
        Args:
            item: Korean vocabulary or grammar to process
            
        Returns:
            Dictionary with processed results
        """
        try:
            logger.debug(f"Processing item: {item}")
            
            # Using the new API style (v1.0.0+)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": item}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=500    # Limit response length
            )
            content = response.choices[0].message.content
            
            # Extract the response content
            result = {
                "item": item,
                "analysis": content,
                "model": self.model
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item '{item}': {str(e)}")
            return {
                "item": item,
                "analysis": f"Error: {str(e)}",
                "model": self.model,
                "error": True
            }
    
    def process_batch(self, items: List[str], batch_size: int = 10, delay: float = 0.5) -> List[Dict]:
        """
        Process a batch of items using the OpenAI API.
        
        Args:
            items: List of vocabulary or grammar items to process
            batch_size: Number of items to process concurrently
            delay: Delay between batches (to avoid rate limits)
            
        Returns:
            List of dictionaries with processed results
        """
        results = []
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(items), batch_size), desc="Processing batches"):
            batch = items[i:i+batch_size]
            batch_results = []
            
            # Process each item in the batch
            for item in batch:
                result = self.process_item(item)
                batch_results.append(result)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            results.extend(batch_results)
            
            # Delay between batches
            if i + batch_size < len(items):
                logger.debug(f"Sleeping for {delay} seconds between batches")
                time.sleep(delay)
        
        logger.info(f"Processed {len(results)} items")
        return results
    
    def process_vocabulary(self, vocabulary: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Process vocabulary items.
        
        Args:
            vocabulary: List of vocabulary items to process
            batch_size: Number of items to process in each batch
            
        Returns:
            List of dictionaries with processed results
        """
        logger.info(f"Processing {len(vocabulary)} vocabulary items in batches of {batch_size}")
        return self.process_batch(vocabulary, batch_size)
    
    def process_grammar(self, grammar: List[tuple], batch_size: int = 5) -> List[Dict]:
        """
        Process grammar items with examples.
        
        Args:
            grammar: List of (grammar_pattern, example) tuples
            batch_size: Number of items to process in each batch
            
        Returns:
            List of dictionaries with processed results
        """
        # Convert grammar tuples to appropriate format for processing
        grammar_items = [f"문법: {pattern}\n예문: {example}" for pattern, example in grammar]
        
        logger.info(f"Processing {len(grammar_items)} grammar items in batches of {batch_size}")
        return self.process_batch(grammar_items, batch_size)


def process_with_openai(data: Dict, batch_size: int = 10) -> Dict:
    """
    Convenience function to process data with OpenAI.
    
    Args:
        data: Dictionary with vocabulary and grammar items
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with processed results
    """
    try:
        processor = OpenAIProcessor()
        
        # Process vocabulary
        vocabulary_results = processor.process_vocabulary(data['vocabulary'], batch_size)
        
        # Process grammar
        grammar_results = processor.process_grammar(data['grammar'], batch_size // 2)
        
        return {
            'vocabulary_results': vocabulary_results,
            'grammar_results': grammar_results
        }
        
    except Exception as e:
        logger.error(f"Error processing with OpenAI: {str(e)}")
        raise 