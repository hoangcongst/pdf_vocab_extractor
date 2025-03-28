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


async def process_with_openai_async(data: Dict[str, Any], batch_size: int = 5) -> Dict[str, Any]:
    """
    Process data asynchronously using OpenAI API.
    Args:
        data (Dict[str, Any]): Data containing vocabulary and/or grammar items
        batch_size (int): Number of items to process in each batch
    Returns:
        Dict[str, Any]: Processed data with OpenAI responses
    """
    processor = OpenAIProcessor()
    
    # Process vocabulary if present
    vocabulary_results = []
    if 'vocabulary' in data:
        vocabulary_results = await processor.process_vocabulary_async(data['vocabulary'], batch_size)
    
    # Process grammar if present
    grammar_results = []
    if 'grammar' in data:
        grammar_results = await processor.process_grammar_async(data['grammar'], batch_size // 2)
    
    return {
        'vocabulary_results': vocabulary_results,
        'grammar_results': grammar_results
    }

def process_with_openai(data: Dict, batch_size: int = 10) -> Dict:
    """
    Synchronous wrapper for async processing.
    """
    return asyncio.run(process_with_openai_async(data, batch_size))

def format_word_analysis(word_data: Dict) -> str:
    """
    Format a word's analysis data into HTML format.
    
    Args:
        word_data: Dictionary containing word analysis data
        
    Returns:
        Formatted HTML string with the analysis
    """
    try:
        output = []
        
        # Add word and meanings
        word = word_data.get('item', '')
        analysis = word_data.get('analysis', {})
        if isinstance(analysis, str):
            # Handle error case where analysis is a string
            return f'<div class="word-analysis error"><h3>{word}</h3><pre>{analysis}</pre></div>'
            
        output.append(f'<div class="word-analysis">')
        output.append(f'<h3 class="word">{word}</h3>')
        
        # Handle meanings
        try:
            meanings = analysis.get('meanings', [])
            if meanings:
                output.append('<div class="meanings">')
                output.append('<h4>Nghĩa:</h4>')
                output.append('<ul>')
                if isinstance(meanings, list):
                    for meaning in meanings:
                        output.append(f'<li>{meaning}</li>')
                elif isinstance(meanings, str):
                    output.append(f'<li>{meanings}</li>')
                output.append('</ul>')
                output.append('</div>')
        except Exception as e:
            logger.warning(f"Error formatting meanings for {word}: {str(e)}")
        
        # Handle examples
        try:
            examples = analysis.get('examples', {})
            if examples:
                output.append('<div class="examples">')
                output.append('<h4>Ví dụ:</h4>')
                output.append('<ul>')
                if isinstance(examples, dict):
                    for meaning_examples in examples.values():
                        if isinstance(meaning_examples, list):
                            for example in meaning_examples:
                                if isinstance(example, dict):
                                    output.append('<li>')
                                    output.append(f'<p class="korean">{example.get("korean", "")}</p>')
                                    output.append(f'<p class="vietnamese">{example.get("vietnamese", "")}</p>')
                                    output.append('</li>')
                elif isinstance(examples, list):
                    for example in examples:
                        if isinstance(example, dict):
                            output.append('<li>')
                            output.append(f'<p class="korean">{example.get("korean", "")}</p>')
                            output.append(f'<p class="vietnamese">{example.get("vietnamese", "")}</p>')
                            output.append('</li>')
                        elif isinstance(example, str):
                            output.append(f'<li>{example}</li>')
                output.append('</ul>')
                output.append('</div>')
        except Exception as e:
            logger.warning(f"Error formatting examples for {word}: {str(e)}")
        
        # Handle memory tip
        try:
            memory_tip = analysis.get('memory_tip')
            if memory_tip:
                output.append('<div class="memory-tip">')
                output.append('<h4>Tip để nhớ từ:</h4>')
                output.append(f'<p>{memory_tip}</p>')
                output.append('</div>')
        except Exception as e:
            logger.warning(f"Error formatting memory tip for {word}: {str(e)}")
        
        # Handle Hanja analysis
        try:
            hanja = analysis.get('hanja_analysis', {})
            if hanja:
                output.append('<div class="hanja-analysis">')
                output.append('<h4>Phân tích Hán tự:</h4>')
                if isinstance(hanja, dict):
                    if explanation := hanja.get('explanation'):
                        output.append(f'<p>{explanation}</p>')
                    if related_words := hanja.get('related_words', []):
                        if isinstance(related_words, list):
                            output.append('<div class="related-words">')
                            output.append('<h5>Từ liên quan:</h5>')
                            output.append('<ul>')
                            for word in related_words:
                                output.append(f'<li>{word}</li>')
                            output.append('</ul>')
                            output.append('</div>')
                elif isinstance(hanja, str):
                    output.append(f'<p>{hanja}</p>')
                output.append('</div>')
        except Exception as e:
            logger.warning(f"Error formatting Hanja analysis for {word}: {str(e)}")
        
        # Handle grammar points
        try:
            grammar = analysis.get('grammar_points', {})
            if grammar:
                output.append('<div class="grammar-points">')
                output.append('<h4>Ngữ pháp:</h4>')
                if isinstance(grammar, dict):
                    output.append('<ul>')
                    if usage := grammar.get('usage'):
                        output.append(f'<li><strong>Cách dùng:</strong> {usage}</li>')
                    if conjugation := grammar.get('conjugation'):
                        output.append(f'<li><strong>Cách chia:</strong> {conjugation}</li>')
                    if formality := grammar.get('formality'):
                        output.append(f'<li><strong>Mức độ trang trọng:</strong> {formality}</li>')
                    output.append('</ul>')
                elif isinstance(grammar, str):
                    output.append(f'<p>{grammar}</p>')
                output.append('</div>')
        except Exception as e:
            logger.warning(f"Error formatting grammar points for {word}: {str(e)}")
        
        output.append('</div>')  # Close word-analysis div
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Error formatting word analysis for {word_data.get('item', '')}: {str(e)}")
        # Return a basic HTML format with the raw data
        return f'<div class="word-analysis error"><h3>{word_data.get("item", "")}</h3><pre>{str(word_data)}</pre></div>'

def format_results_to_text(results: List[Dict]) -> str:
    """
    Format a list of word analysis results into HTML.
    
    Args:
        results: List of dictionaries containing word analysis
        
    Returns:
        Formatted HTML string with all analyses
    """
    if not results:
        return ""
        
    output = ['<div class="vocabulary-results">']
    
    for result in results:
        if isinstance(result.get('analysis'), dict):
            # If analysis is already a dictionary, use it directly
            word_data = {
                'item': result['item'],
                **result['analysis']
            }
            formatted_analysis = format_word_analysis(word_data)
        else:
            # If analysis is a string, create a simple format
            formatted_analysis = f'<div class="word-analysis error"><h3>{result["item"]}</h3><pre>{result["analysis"]}</pre></div>'
        
        output.append(formatted_analysis)
    
    output.append('</div>')  # Close vocabulary-results div
    return "\n".join(output) 