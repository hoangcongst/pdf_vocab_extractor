# Korean Language Extraction and Analysis Tool

This project extracts Korean vocabulary and grammar from PDF files, processes them in batches using GPT-4o-mini for efficiency, and exports the results to an Excel file for learners at TOPIK 3-4 (intermediate) level.

## Requirements
- Extract Korean vocabulary from PDF files and convert them to their original forms (e.g., handling conjugations or variations).
- Identify and extract intermediate-level (TOPIK 3-4) Korean grammar structures from the PDFs.
- Analyze extracted vocabulary and grammar in batches using GPT-4o-mini with the provided prompt:
```
Bạn là từ điển AI dịch từ tiếng Hàn sang tiếng Việt, tôi gửi bạn 1 từ tiếng Hàn, bạn sẽ đưa ra các nghĩa cho tôi và kèm ví dụ, tip để tôi có thể nhớ được từ.                           
- Nếu là từ Hán Hàn, hãy phân tích từ tiếng Hán cấu tạo nên từ đó. Ví dụ: 방법. 방 là phương, 법 là pháp nên 방법 là phương pháp.
Nêu 1 số ví dụ từ Hán Hàn được cấu tạo từ từ cấu thành nên từ gốc.                             
- Nếu input là cấu trúc ngữ pháp hoặc câu đầy đủ thì Phân tích các cấu trúc ngữ pháp nếu có xuất hiện dành cho trình độ topik trung cấp 1, 2
- Không gửi nội dung thừa, không phiên âm
```
- Deduplicate and normalize words using GPT-4o-mini batch processing with the prompt:
```
Chuyển từ về dạng nguyên thể và xóa từ trùng lặp
```
- Export the processed vocabulary and grammar results into an Excel file.

## Quick Start

Use our one-command script to extract, deduplicate, and process vocabulary:

```bash
./extract_vocab.sh your_pdf_file.pdf output_file.xlsx
```

This script will:
1. Extract all Korean vocabulary from the PDF file
2. Process the vocabulary in batches of 200 words with GPT-4o-mini to:
   - Convert words to their base forms
   - Remove duplicates
3. Process each word with GPT-4o-mini for full translation and analysis
4. Save all results to an Excel file

## Tech Stack
- **Python**: Core programming language for PDF processing, batch API calls, and Excel export.
- Libraries (to be finalized by the dev team):
  - PDF parsing (e.g., PyPDF2 or pdfplumber).
  - GPT-4o-mini API integration with batch processing support (e.g., OpenAI Python client).
  - Excel file handling (e.g., pandas or openpyxl).

## Milestones
1. **PDF Parsing Setup**: Build a system to extract raw text (vocabulary and sentences) from PDF files.
2. **Vocabulary & Grammar Detection**: Add logic to identify Korean vocabulary (original forms) and TOPIK 3-4 grammar structures from the extracted text.
3. **Batch Processing with GPT-4o-mini**: Integrate GPT-4o-mini to process vocabulary and grammar in batches for faster performance.
4. **Result Formatting**: Structure the batched analysis output (meanings, examples, Hanja breakdowns, grammar explanations) for clarity.
5. **Excel Export**: Export the final processed data into a clean, organized Excel file.
6. **Batch Deduplication**: Add functionality to deduplicate and normalize words in large batches for efficiency.

## Advanced Usage

For more control over the extraction process, you can use the Python module directly:

```bash
python -m src.main your_pdf_file.pdf --output result.xlsx --dedupe-batch --dedupe-batch-size 200
```

### Available Options

- `--dedupe-batch`: Enable batch deduplication and normalization
- `--dedupe-batch-size`: Set number of words per batch for deduplication (default: 200)
- `--batch-size`: Set batch size for translation processing (default: 10)
- `--method`: Choose PDF extraction method (`pdfplumber` or `pypdf2`, default: `pdfplumber`)
- `--mecab`: Use Mecab tokenizer instead of Okt
- `--skip-gpt`: Skip GPT processing (useful for testing)
- `--limit`: Process only a limited number of words (for testing)
- `--verbose`: Enable verbose logging