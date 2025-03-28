import asyncio
import time
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.gpt_integration.openai_client import OpenAIProcessor
import json
import logging
from typing import List, Dict

# Configure logging to show all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug logging for all loggers
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# Test data - 20 Korean words
test_words = [
    "가게", "가격", "가구", "가능성", "가다가",
    "가려서", "가로", "가서", "가수", "가슴",
    "가야", "가운", "가장", "가져오는", "가족",
    "각각", "각자", "간단히", "갈수록", "감기"
]

# Test grammar patterns
test_grammar = [
    ("~(으)ㄹ 수 있다", "나는 한국어를 말할 수 있어요."),
    ("~아/어/여야 하다", "내일 일찍 일어나야 해요."),
    ("~(으)ㄹ 때", "한국에 갔을 때 많이 배웠어요."),
    ("~(으)면서", "음악을 들으면서 공부해요.")
]

def print_result(result: Dict, indent: str = ""):
    """Helper function to print results in a readable format"""
    print(f"{indent}Item: {result['item']}")
    
    if isinstance(result['analysis'], dict):
        analysis = result['analysis']
        
        # Print meanings
        if analysis.get('meanings'):
            print(f"{indent}Meanings:")
            for i, meaning in enumerate(analysis['meanings'], 1):
                print(f"{indent}  {i}. {meaning}")
        
        # Print examples
        if analysis.get('examples'):
            print(f"{indent}Examples:")
            for meaning_key, examples in analysis['examples'].items():
                for example in examples:
                    print(f"{indent}  - {example.get('korean', '')}")
                    print(f"{indent}    {example.get('vietnamese', '')}")
        
        # Print memory tip
        if analysis.get('memory_tip'):
            print(f"{indent}Memory Tip: {analysis['memory_tip']}")
        
        # Print grammar points
        if analysis.get('grammar_points'):
            grammar = analysis['grammar_points']
            print(f"{indent}Grammar:")
            for key, value in grammar.items():
                if value:
                    print(f"{indent}  {key}: {value}")
    else:
        print(f"{indent}Analysis: {result['analysis']}")
    
    if result.get('error'):
        print(f"{indent}Error occurred during processing")
    print(f"{indent}" + "-" * 40)

async def test_batch_async():
    """Test async batch processing"""
    logger.info("Testing async batch processing...")
    processor = OpenAIProcessor()
    
    # Test different batch sizes
    batch_sizes = [5, 10]
    for batch_size in batch_sizes:
        start_time = time.time()
        logger.info(f"Processing with batch size {batch_size}...")
        
        # Process vocabulary
        vocab_results = await processor.process_vocabulary_async(test_words[:10], batch_size=batch_size)
        print(f"\nVocabulary Results (batch_size={batch_size}):")
        for result in vocab_results:
            print_result(result, indent="  ")
        
        # Process grammar
        grammar_results = await processor.process_grammar_async(test_grammar[:2], batch_size=batch_size)
        print(f"\nGrammar Results (batch_size={batch_size}):")
        for result in grammar_results:
            print_result(result, indent="  ")
        
        end_time = time.time()
        logger.info(f"Batch size {batch_size} completed in {end_time - start_time:.2f} seconds")

async def test_batch_sync():
    """Test synchronous batch processing"""
    logger.info("Testing synchronous batch processing...")
    processor = OpenAIProcessor()
    
    # Process with default batch size
    start_time = time.time()
    # Use async method directly in async context
    results = await processor.process_vocabulary_async(test_words[:5], batch_size=5)
    
    print("\nSync Processing Results:")
    for result in results:
        print_result(result, indent="  ")
    
    end_time = time.time()
    logger.info(f"Sync processing completed in {end_time - start_time:.2f} seconds")

async def test_concurrent_batches():
    """Test processing multiple batches concurrently"""
    logger.info("Testing concurrent batch processing...")
    processor = OpenAIProcessor()
    start_time = time.time()
    
    # Process vocabulary and grammar concurrently
    vocab_task = processor.process_vocabulary_async(test_words[:5], batch_size=5)
    grammar_task = processor.process_grammar_async(test_grammar[:2], batch_size=2)
    
    vocab_results, grammar_results = await asyncio.gather(vocab_task, grammar_task)
    
    print("\nConcurrent Processing Results:")
    print("\nVocabulary:")
    for result in vocab_results:
        print_result(result, indent="  ")
    
    print("\nGrammar:")
    for result in grammar_results:
        print_result(result, indent="  ")
    
    end_time = time.time()
    logger.info(f"Concurrent processing completed in {end_time - start_time:.2f} seconds")

async def main():
    """Main test function"""
    try:
        # Test async batch processing
        await test_batch_async()
        
        # Test sync batch processing
        await test_batch_sync()
        
        # Test concurrent processing
        await test_concurrent_batches()
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 