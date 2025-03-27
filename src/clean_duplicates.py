import pandas as pd
import re
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import KoNLPy
try:
    from konlpy.tag import Okt
    KONLPY_AVAILABLE = True
    okt = Okt()
    logger.info("KoNLPy Okt tokenizer loaded successfully")
except ImportError:
    logger.warning("KoNLPy not available. Using regex-based fallback methods.")
    KONLPY_AVAILABLE = False

def normalize_word_konlpy(word: str) -> str:
    """
    Normalize a Korean word using KoNLPy Okt.
    
    Args:
        word: Korean word to normalize
        
    Returns:
        Normalized word
    """
    if not KONLPY_AVAILABLE:
        return word
        
    try:
        # Try to normalize the word
        norm = okt.normalize(word)
        
        # Get POS information with normalization
        pos_result = okt.pos(word, norm=True)
        
        # If it's a verb or adjective, try to get the stem
        if pos_result and len(pos_result) > 0:
            if pos_result[0][1].startswith('V') or pos_result[0][1].startswith('J'):
                # Return the normalized form
                return pos_result[0][0]
                
        return norm
        
    except:
        # If anything fails, return the original word
        return word

def normalize_word_regex(word: str) -> str:
    """
    Normalize a Korean word using regex patterns (fallback).
    
    Args:
        word: Korean word to normalize
        
    Returns:
        Normalized word
    """
    if not isinstance(word, str):
        return str(word)
        
    # Remove common Korean particles and endings
    # First try removing longer particles
    base = re.sub(r'(으로|에서|에게|부터|까지|처럼|마다|보다)$', '', word)
    # Then try removing single character particles
    base = re.sub(r'[과와은는이가을를에의도]$', '', base)
    
    return base

def clean_duplicates(file_path, output_file=None):
    """Clean and deduplicate Korean vocabulary from an Excel file."""
    # Read the Excel file
    logger.info(f"Reading file: {file_path}")
    df = pd.read_excel(file_path)
    
    # Extract words
    words = df['Word'].tolist()
    logger.info(f"Extracted {len(words)} words")
    
    # Use KoNLPy or regex to normalize and deduplicate
    unique_words = deduplicate_words(words)
    logger.info(f"Reduced to {len(unique_words)} unique words")
    
    # Create the output dataframe
    output_data = []
    for word in unique_words:
        output_data.append({
            'Word': word,
            'Analysis': '',  # Empty analysis for now
            'Model': 'KoNLPy Normalization' if KONLPY_AVAILABLE else 'Regex Deduplication',
            'Has Error': 'No'
        })
    
    # Create new DataFrame and save to Excel
    new_df = pd.DataFrame(output_data)
    
    # Determine output file name
    if output_file is None:
        output_file = file_path.replace('.xlsx', '_cleaned.xlsx')
    
    # Save to Excel
    new_df.to_excel(output_file, index=False)
    logger.info(f"Cleaned data saved to {output_file}")
    return output_file

def deduplicate_words(words: List[str]) -> List[str]:
    """
    Deduplicate a list of Korean words by normalizing them.
    
    Args:
        words: List of Korean words
        
    Returns:
        List of deduplicated words
    """
    # Dictionary to store normalized form -> original form mapping
    word_forms = {}
    
    # Process each word
    for word in words:
        if not word or not isinstance(word, str):
            continue
            
        # Normalize the word using KoNLPy or regex
        if KONLPY_AVAILABLE:
            base = normalize_word_konlpy(word)
        else:
            base = normalize_word_regex(word)
            
        # Skip very short words
        if len(base) < 2:
            continue
            
        # Store the mapping
        if base not in word_forms:
            word_forms[base] = []
        word_forms[base].append(word)
    
    # Get representative words (shortest form for each base)
    unique_words = []
    for base, forms in word_forms.items():
        try:
            # Find the shortest form
            representative = min(forms, key=len)
            unique_words.append(representative)
        except Exception as e:
            logger.error(f"Error finding representative for base '{base}': {str(e)}")
    
    return sorted(unique_words)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        clean_duplicates(input_file, output_file)
    else:
        print("Usage: python clean_duplicates.py input.xlsx [output.xlsx]") 