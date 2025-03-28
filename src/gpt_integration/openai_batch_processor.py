"""
OpenAI Batch Text Processor Module

This module handles processing lists of Korean words with GPT-4o mini to:
1. Convert words to their base forms 
2. Remove duplicates
"""

import os
import logging
import time
from typing import List, Dict, Any
import json

import openai
from tqdm import tqdm
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load API key from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")

# Extract project ID from API key if it's a project key
PROJECT_ID = None
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-proj-"):
    # Project ID is the first part after "sk-proj-" up to the first underscore
    PROJECT_ID = "org-" + OPENAI_API_KEY.split("sk-proj-")[1].split("_")[0]

# Check if API key is available
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables.")


class BatchDeduplicator:
    """Class to process batches of text using OpenAI API to deduplicate and lemmatize words."""
    
    # The standard prompt for normalizing and deduplicating words
    DEDUPE_PROMPT = """
    Chuyển từ về dạng nguyên thể và xóa từ trùng lặp.
    
    Quy tắc xử lý:
    1. Với mỗi từ, chuyển về dạng nguyên thể (lemmatize)
    2. Loại bỏ các phần tử trùng lặp
    3. Trả về danh sách các từ đã xử lý theo định dạng JSON
    4. Chỉ trả về mảng JSON thuần túy, không thêm giải thích hay định dạng khác
    
    Kết quả cần là 1 mảng JSON các chuỗi, ví dụ: ["từ1", "từ2", "từ3"]
    """
    
    def __init__(self, api_key=None, model=None, prompt=None):
        """
        Initialize the batch processor.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model to use (defaults to environment variable or gpt-4)
            prompt: System prompt template (defaults to Korean deduplication)
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            raise ValueError("OpenAI API key is required")
        
        self.model = model or OPENAI_MODEL
        logger.info(f"Using OpenAI model: {self.model}")
        
        self.system_prompt = prompt or self.DEDUPE_PROMPT
        
        # Initialize OpenAI client with organization ID
        client_args = {"api_key": self.api_key}
        if OPENAI_ORG_ID:
            client_args["organization"] = OPENAI_ORG_ID
            logger.info(f"Using organization ID: {OPENAI_ORG_ID}")
        
        self.client = openai.OpenAI(**client_args)
    
    def process_batch(self, words: List[str], max_retries=3) -> List[str]:
        """
        Process a batch of words using the OpenAI API to normalize and deduplicate.
        
        Args:
            words: List of words to process
            max_retries: Maximum number of retries on error
            
        Returns:
            List of deduplicated and normalized words
        """
        if not words:
            return []
            
        try:
            logger.debug(f"Processing batch of {len(words)} words")
            
            # Convert list to JSON for API call
            words_json = json.dumps(words, ensure_ascii=False)
            
            # Create the API call
            for attempt in range(max_retries):
                try:
                    # Using the new client API style (v1.0.0+)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": words_json}
                        ],
                        temperature=0.1,  # Low temperature for consistent results
                        max_tokens=4000   # Allow enough tokens for response
                    )
                    result_text = response.choices[0].message.content.strip()
                    
                    # Parse the JSON response, handling potential formatting issues
                    try:
                        # Try to parse as-is first
                        processed_words = json.loads(result_text)
                        
                        # Validate that we got a list of strings
                        if isinstance(processed_words, list) and all(isinstance(item, str) for item in processed_words):
                            return processed_words
                        else:
                            logger.warning(f"Response was not a list of strings, retrying. Got: {type(processed_words)}")
                            continue
                            
                    except json.JSONDecodeError:
                        # Try to extract JSON if it's wrapped in markdown or other text
                        import re
                        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                        
                        if json_match:
                            try:
                                processed_words = json.loads(json_match.group(0))
                                if isinstance(processed_words, list):
                                    return processed_words
                            except:
                                pass
                                
                        logger.warning(f"Failed to parse JSON response, retrying. Response: {result_text[:100]}...")
                        
                except Exception as e:
                    logger.warning(f"API call attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise
                        
            # If we get here, all retries failed
            logger.error("All retries failed to get valid response from OpenAI API")
            return words  # Return original list as fallback
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return words  # Return original list as fallback
    
    def process_all_words(self, all_words: List[str], batch_size: int = 200, delay: float = 1.0) -> List[str]:
        """
        Process all words in batches, normalizing and deduplicating.
        
        Args:
            all_words: Complete list of words to process
            batch_size: Number of words to process in each batch
            delay: Delay between batches (to avoid rate limits)
            
        Returns:
            List of deduplicated and normalized words
        """
        if not all_words:
            return []
            
        unique_words = set()
        logger.info(f"Processing {len(all_words)} words in batches of {batch_size}")
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(all_words), batch_size), desc="Processing word batches"):
            batch = all_words[i:min(i+batch_size, len(all_words))]
            
            # Process the batch
            processed_batch = self.process_batch(batch)
            
            # Add to set for automatic deduplication
            unique_words.update(processed_batch)
            
            # Delay between batches
            if i + batch_size < len(all_words):
                logger.debug(f"Sleeping for {delay} seconds between batches")
                time.sleep(delay)
        
        # Convert back to sorted list
        result = sorted(list(unique_words))
        logger.info(f"Processed {len(all_words)} words into {len(result)} unique normalized words")
        return result


def process_and_deduplicate(words: List[str], batch_size: int = 200) -> List[str]:
    """
    Convenience function to process and deduplicate words with OpenAI.
    
    Args:
        words: List of words to process
        batch_size: Batch size for processing
        
    Returns:
        List of deduplicated and normalized words
    """
    try:
        processor = BatchDeduplicator()
        return processor.process_all_words(words, batch_size)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise 