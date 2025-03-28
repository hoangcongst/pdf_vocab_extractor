- Extract Korean vocabulary from PDF files and convert them to their original forms (e.g., handling conjugations or variations).
- Identify and extract intermediate-level Korean grammar structures from the PDFs.
- Analyze extracted vocabulary and grammar in batches using GPT-4o-mini with the provided prompt:
```
Bạn là từ điển AI dịch từ tiếng Hàn sang tiếng Việt, tôi gửi bạn 1 từ tiếng Hàn, bạn sẽ đưa ra các nghĩa cho tôi và kèm ví dụ, tip để tôi có thể nhớ được từ.                           
- Nếu là từ Hán Hàn, hãy phân tích từ tiếng Hán cấu tạo nên từ đó. Ví dụ: 방법. 방 là phương, 법 là pháp nên 방법 là phương pháp.
Nêu 1 số ví dụ từ Hán Hàn được cấu tạo từ từ cấu thành nên từ gốc.                             
- Nếu input là cấu trúc ngữ pháp hoặc câu đầy đủ thì Phân tích các cấu trúc ngữ pháp nếu có xuất hiện dành cho trình độ topik trung cấp 1, 2
- Không gửi nội dung thừa, không phiên âm
```
- Export the processed vocabulary and grammar results into CSV files (vocabulary.csv and grammar.csv).