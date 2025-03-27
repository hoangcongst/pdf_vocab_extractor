#!/bin/bash
# extract_vocab.sh - Korean vocabulary extraction with KoNLPy normalization

# Check if a PDF file was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <pdf_file> [output_file.xlsx]"
    echo "Example: $0 pdf/topik_book.pdf topik_vocab.xlsx"
    exit 1
fi

PDF_FILE=$1
TEMP_OUTPUT="temp_output_$(date +%Y%m%d_%H%M%S).xlsx"
OUTPUT_FILE=${2:-"output_$(date +%Y%m%d_%H%M%S).xlsx"}

echo "Starting Korean vocabulary extraction with KoNLPy normalization..."
echo "PDF File: $PDF_FILE"
echo "Final Output File: $OUTPUT_FILE"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Step 1: Extract vocabulary to temporary file (skipping GPT analysis)
echo "Step 1: Extracting vocabulary from PDF..."
python -m src.main "$PDF_FILE" --output "$TEMP_OUTPUT" --skip-gpt --verbose --mecab

# Check if the command executed successfully
if [ $? -ne 0 ]; then
    echo "❌ Error during vocabulary extraction. Check the logs for details."
    exit 1
fi

# Step 2: Process with KoNLPy-based deduplication
echo "Step 2: Normalizing and deduplicating words with KoNLPy..."
python -c "from src.clean_duplicates import clean_duplicates; clean_duplicates('$TEMP_OUTPUT', '$OUTPUT_FILE')"

# Check if the command executed successfully
if [ $? -ne 0 ]; then
    echo "❌ Error during deduplication. Check the logs for details."
    exit 1
fi

# Cleanup temporary file
echo "Cleaning up temporary files..."
rm -f "$TEMP_OUTPUT"

echo "✅ Processing completed successfully!"
echo "📊 Results saved to: $OUTPUT_FILE"

# Provide a summary of the results
echo "📝 Summary of results:"
python -c "import pandas as pd; df = pd.read_excel('$OUTPUT_FILE'); print(f'Total unique words: {len(df)}')" 