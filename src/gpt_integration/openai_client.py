"""
OpenAI Integration Module

This module handles integration with OpenAI API for processing Korean vocabulary and grammar.
"""

import os
import logging
import time
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import json

from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import httpx


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
    logger.info(f"Extracted project ID from API key: {PROJECT_ID}")

# Check if API key is available
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found in environment variables.")

class OpenAIProcessor:
    """Class to process text using OpenAI API."""
    
    # The standard prompt for Korean-Vietnamese translation and analysis
    DEFAULT_PROMPT = """
    Bạn là từ điển AI dịch từ tiếng Hàn sang tiếng Việt. Tôi sẽ gửi cho bạn một danh sách các từ tiếng Hàn, mỗi từ trên một dòng.
    Với mỗi từ, hãy phân tích chi tiết theo các mục sau:
    
    1. Nghĩa của từ:
       - Liệt kê đầy đủ các nghĩa khác nhau của từ
       - Nếu có nhiều nghĩa, đánh số từng nghĩa
       
    2. Ví dụ và cách sử dụng:
       - Cung cấp 2-3 ví dụ tiêu biểu cho từng nghĩa
       - Mỗi ví dụ gồm câu tiếng Hàn và nghĩa tiếng Việt
       
    3. Tip để nhớ từ:
       - Đưa ra các mẹo ghi nhớ dựa trên hình ảnh hoặc tình huống thực tế
       - Liên hệ với những từ hoặc khái niệm tương tự để dễ nhớ
       
    4. Phân tích Hán tự (nếu có):
       - Giải thích ý nghĩa của từng chữ Hán
       - Liệt kê một số từ Hán Hàn khác được tạo từ các chữ Hán này
       
    5. Cấu trúc ngữ pháp (nếu là cấu trúc ngữ pháp):
       - Giải thích cách sử dụng và quy tắc ngữ pháp
       - Các dạng biến thể và cách chia
       - Mức độ trang trọng/thân mật
    
    Hãy trả lời theo định dạng JSON như sau:
    {
        "words": [
            {
                "word": "từ gốc",
                "meanings": ["nghĩa 1", "nghĩa 2", ...],
                "examples": {
                    "meaning1": [
                        {
                            "korean": "câu ví dụ tiếng Hàn",
                            "vietnamese": "nghĩa tiếng Việt"
                        },
                        ...
                    ],
                    "meaning2": [...]
                },
                "memory_tip": "mẹo để nhớ từ",
                "hanja_analysis": {
                    "explanation": "giải thích ý nghĩa Hán tự",
                    "related_words": ["từ liên quan 1", "từ liên quan 2"]
                },
                "grammar_points": {
                    "usage": "cách sử dụng",
                    "conjugation": "cách chia",
                    "formality": "mức độ trang trọng"
                }
            },
            ...
        ]
    }
    """
    
    def __init__(self, api_key=None, model=None, prompt=None):
        """
        Initialize the OpenAI processor.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model to use (defaults to environment variable or gpt-4)
            prompt: System prompt template (defaults to Korean-Vietnamese translation)
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            logger.error("No OpenAI API key provided")
            raise ValueError("OpenAI API key is required")
        
        self.model = model or OPENAI_MODEL
        logger.info(f"Using OpenAI model: {self.model}")
        
        self.system_prompt = prompt or self.DEFAULT_PROMPT
        
        # Initialize OpenAI client with organization ID
        client_args = {"api_key": self.api_key}
        if OPENAI_ORG_ID:
            client_args["organization"] = OPENAI_ORG_ID
            logger.info(f"Using organization ID: {OPENAI_ORG_ID}")
        
        self.client = OpenAI(**client_args)
    
    def process_batch_items(self, items: List[str]) -> List[Dict]:
        """
        Process multiple items in a single API request.
        
        Args:
            items: List of Korean vocabulary or grammar items to process
            
        Returns:
            List of dictionaries with processed results
        """
        try:
            # Format items as a numbered list
            items_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
            logger.debug(f"Processing batch of {len(items)} items")
            
            # Make API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": items_text}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=4000,   # Increased token limit for detailed responses
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            try:
                # Parse JSON response
                content = response.choices[0].message.content
                parsed_response = json.loads(content)
                
                # Map response back to items
                results = []
                for i, item in enumerate(items):
                    if i < len(parsed_response.get("words", [])):
                        word_data = parsed_response["words"][i]
                        result = {
                            "item": item,
                            "analysis": {
                                "meanings": word_data.get("meanings", []),
                                "examples": word_data.get("examples", {}),
                                "memory_tip": word_data.get("memory_tip", ""),
                                "hanja_analysis": word_data.get("hanja_analysis", {}),
                                "grammar_points": word_data.get("grammar_points", {})
                            },
                            "model": self.model
                        }
                    else:
                        # Handle case where response has fewer items than input
                        result = {
                            "item": item,
                            "analysis": "Error: No analysis provided in response",
                            "model": self.model,
                            "error": True
                        }
                    results.append(result)
                
                return results
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Return error results for all items
                return [{
                    "item": item,
                    "analysis": f"Error: Failed to parse response - {str(e)}",
                    "model": self.model,
                    "error": True
                } for item in items]
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Return error results for all items
            return [{
                "item": item,
                "analysis": f"Error: {str(e)}",
                "model": self.model,
                "error": True
            } for item in items]
    
    def process_batch(self, items: List[str], batch_size: int = 10, delay: float = 0.5) -> List[Dict]:
        """
        Process items in batches using the OpenAI API.
        
        Args:
            items: List of vocabulary or grammar items to process
            batch_size: Number of items to process in each API request
            delay: Delay between batches (to avoid rate limits)
            
        Returns:
            List of dictionaries with processed results
        """
        results = []
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(items), batch_size), desc="Processing batches"):
            batch = items[i:i+batch_size]
            
            # Process the entire batch in one API call
            batch_results = self.process_batch_items(batch)
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

    async def process_batch_items_async(self, items: List[str]) -> List[Dict]:
        """
        Process multiple items in a single API request asynchronously.
        
        Args:
            items: List of Korean vocabulary or grammar items to process
            
        Returns:
            List of dictionaries with processed results
        """
        try:
            # Format items as a numbered list
            items_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
            logger.debug(f"Processing batch of {len(items)} items")
            
            # Initialize async client if not already done
            if not hasattr(self, 'async_client'):
                client_args = {"api_key": self.api_key}
                if OPENAI_ORG_ID:
                    client_args["organization"] = OPENAI_ORG_ID
                self.async_client = AsyncOpenAI(**client_args)
            
            # Make async API request
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": items_text}
                ],
                temperature=0.3,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            try:
                # Parse JSON response
                content = response.choices[0].message.content
                parsed_response = json.loads(content)
                
                # Map response back to items
                results = []
                for i, item in enumerate(items):
                    if i < len(parsed_response.get("words", [])):
                        word_data = parsed_response["words"][i]
                        result = {
                            "item": item,
                            "analysis": {
                                "meanings": word_data.get("meanings", []),
                                "examples": word_data.get("examples", {}),
                                "memory_tip": word_data.get("memory_tip", ""),
                                "hanja_analysis": word_data.get("hanja_analysis", {}),
                                "grammar_points": word_data.get("grammar_points", {})
                            },
                            "model": self.model
                        }
                    else:
                        result = {
                            "item": item,
                            "analysis": "Error: No analysis provided in response",
                            "model": self.model,
                            "error": True
                        }
                    results.append(result)
                
                return results
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return [{
                    "item": item,
                    "analysis": f"Error: Failed to parse response - {str(e)}",
                    "model": self.model,
                    "error": True
                } for item in items]
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return [{
                "item": item,
                "analysis": f"Error: {str(e)}",
                "model": self.model,
                "error": True
            } for item in items]

    async def process_batch_async(self, items: List[str], batch_size: int = 10, delay: float = 0.5) -> List[Dict]:
        """
        Process items in batches using the OpenAI API asynchronously.
        
        Args:
            items: List of vocabulary or grammar items to process
            batch_size: Number of items to process in each API request
            delay: Delay between batches (to avoid rate limits)
            
        Returns:
            List of dictionaries with processed results
        """
        results = []
        tasks = []
        
        # Create tasks for each batch
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            if i > 0:  # Add delay between batches
                await asyncio.sleep(delay)
            tasks.append(self.process_batch_items_async(batch))
        
        # Process all batches concurrently with progress bar
        batch_results = await tqdm_asyncio.gather(*tasks, desc="Processing batches")
        
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
        
        logger.info(f"Processed {len(results)} items")
        return results

    def process_batch(self, items: List[str], batch_size: int = 10, delay: float = 0.5) -> List[Dict]:
        """
        Synchronous wrapper for async batch processing.
        """
        return asyncio.run(self.process_batch_async(items, batch_size, delay))

    async def process_vocabulary_async(self, vocabulary: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Process vocabulary items asynchronously.
        """
        logger.info(f"Processing {len(vocabulary)} vocabulary items in batches of {batch_size}")
        return await self.process_batch_async(vocabulary, batch_size)

    async def process_grammar_async(self, grammar: List[tuple], batch_size: int = 5) -> List[Dict]:
        """
        Process grammar items with examples asynchronously.
        """
        grammar_items = [f"문법: {pattern}\n예문: {example}" for pattern, example in grammar]
        logger.info(f"Processing {len(grammar_items)} grammar items in batches of {batch_size}")
        return await self.process_batch_async(grammar_items, batch_size)

    def process_vocabulary(self, vocabulary: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Synchronous wrapper for async vocabulary processing.
        """
        return asyncio.run(self.process_vocabulary_async(vocabulary, batch_size))

    def process_grammar(self, grammar: List[tuple], batch_size: int = 5) -> List[Dict]:
        """
        Synchronous wrapper for async grammar processing.
        """
        return asyncio.run(self.process_grammar_async(grammar, batch_size))


async def process_with_openai_async(data: Dict, batch_size: int = 10) -> Dict:
    """
    Async convenience function to process data with OpenAI.
    """
    try:
        processor = OpenAIProcessor()
        
        # Process vocabulary and grammar concurrently
        vocabulary_task = processor.process_vocabulary_async(data['vocabulary'], batch_size)
        grammar_task = processor.process_grammar_async(data['grammar'], batch_size // 2)
        
        vocabulary_results, grammar_results = await asyncio.gather(vocabulary_task, grammar_task)
        
        return {
            'vocabulary_results': vocabulary_results,
            'grammar_results': grammar_results
        }
        
    except Exception as e:
        logger.error(f"Error processing with OpenAI: {str(e)}")
        raise

def process_with_openai(data: Dict, batch_size: int = 10) -> Dict:
    """
    Synchronous wrapper for async processing.
    """
    return asyncio.run(process_with_openai_async(data, batch_size))

def format_word_analysis(word_data: Dict) -> str:
    """
    Format a word's analysis data into a readable text format.
    
    Args:
        word_data: Dictionary containing word analysis data
        
    Returns:
        Formatted string with the analysis
    """
    try:
        output = []
        
        # Add word and meanings
        word = word_data.get('word', '')
        output.append(f"Từ: {word}")
        
        # Handle meanings
        try:
            if meanings := word_data.get('meanings', []):
                if isinstance(meanings, list):
                    output.append("\nNghĩa:")
                    for i, meaning in enumerate(meanings, 1):
                        output.append(f"{i}. {meaning}")
                elif isinstance(meanings, str):
                    output.append("\nNghĩa:")
                    output.append(f"1. {meanings}")
        except Exception as e:
            logger.warning(f"Error formatting meanings: {str(e)}")
        
        # Handle examples
        try:
            if examples := word_data.get('examples'):
                output.append("\nVí dụ:")
                if isinstance(examples, dict):
                    # Handle dictionary format (from test data)
                    for meaning_examples in examples.values():
                        if isinstance(meaning_examples, list):
                            for example in meaning_examples:
                                if isinstance(example, dict):
                                    output.append(f"- {example.get('korean', '')}")
                                    output.append(f"  {example.get('vietnamese', '')}")
                elif isinstance(examples, list):
                    # Handle list format (from OpenAI response)
                    for example in examples:
                        if isinstance(example, dict):
                            output.append(f"- {example.get('korean', '')}")
                            output.append(f"  {example.get('vietnamese', '')}")
                        elif isinstance(example, str):
                            output.append(f"- {example}")
        except Exception as e:
            logger.warning(f"Error formatting examples: {str(e)}")
        
        # Handle memory tip
        try:
            if memory_tip := word_data.get('memory_tip'):
                output.append(f"\nTip để nhớ từ:")
                output.append(str(memory_tip))
        except Exception as e:
            logger.warning(f"Error formatting memory tip: {str(e)}")
        
        # Handle Hanja analysis
        try:
            if hanja := word_data.get('hanja_analysis'):
                output.append("\nPhân tích Hán tự:")
                if isinstance(hanja, dict):
                    if explanation := hanja.get('explanation'):
                        output.append(str(explanation))
                    if related_words := hanja.get('related_words', []):
                        if isinstance(related_words, list):
                            output.append("\nTừ liên quan:")
                            for word in related_words:
                                output.append(f"- {word}")
                elif isinstance(hanja, str):
                    output.append(hanja)
        except Exception as e:
            logger.warning(f"Error formatting Hanja analysis: {str(e)}")
        
        # Handle grammar points
        try:
            if grammar := word_data.get('grammar_points'):
                output.append("\nNgữ pháp:")
                if isinstance(grammar, dict):
                    if usage := grammar.get('usage'):
                        output.append(f"Cách dùng: {usage}")
                    if conjugation := grammar.get('conjugation'):
                        output.append(f"Cách chia: {conjugation}")
                    if formality := grammar.get('formality'):
                        output.append(f"Mức độ trang trọng: {formality}")
                elif isinstance(grammar, str):
                    output.append(grammar)
        except Exception as e:
            logger.warning(f"Error formatting grammar points: {str(e)}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Error formatting word analysis: {str(e)}")
        # Return a basic format with the raw data
        return f"Từ: {word_data.get('item', '')}\n\nPhân tích:\n{str(word_data)}"

def format_results_to_text(results: List[Dict]) -> str:
    """
    Format a list of word analysis results into readable text.
    
    Args:
        results: List of dictionaries containing word analysis
        
    Returns:
        Formatted string with all analyses
    """
    if not results:
        return ""
        
    output = []
    
    for result in results:
        if isinstance(result.get('analysis'), dict):
            # If analysis is already a dictionary, use it directly
            word_data = result['analysis']
            formatted_analysis = format_word_analysis(word_data)
        else:
            # If analysis is a string, create a simple format
            formatted_analysis = f"Từ: {result['item']}\n\nPhân tích:\n{result['analysis']}"
        
        output.append(formatted_analysis)
        output.append("\n" + "="*40 + "\n")  # Separator between words
    
    return "\n".join(output) 