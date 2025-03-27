"""
Sample Data Generation for Testing

This script generates a sample dataset to test the application without
requiring KoNLPy or OpenAI API calls.
"""

import logging
import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.excel_exporter import export_to_excel


# Sample Korean vocabulary
SAMPLE_VOCABULARY = [
    "한국어",
    "공부하다",
    "어려워요",
    "안녕하세요",
    "감사합니다",
    "미안합니다",
    "책",
    "학교",
    "선생님",
    "친구"
]

# Sample grammar patterns with examples
SAMPLE_GRAMMAR = [
    ("(은|는) 것 같다", "한국어는 재미있는 것 같아요."),
    ("(으)?ㄹ 수 있다", "저는 한국어를 할 수 있어요."),
    ("(아|어|여) 보다", "한국 음식을 먹어 봤어요?"),
    ("기 때문에", "바빠서 못 갔어요."),
    ("(으)?면 안 되다", "여기서 담배를 피우면 안 돼요.")
]

# Sample GPT responses for vocabulary
SAMPLE_VOCAB_RESULTS = [
    {
        "item": "한국어",
        "analysis": "한국어 (Tiếng Hàn)\n- Nghĩa: Tiếng Hàn Quốc\n- Phân tích Hán-Hàn: 한국 (Hàn Quốc) + 어 (ngữ/tiếng)\n- Ví dụ: 저는 한국어를 공부해요 (Tôi học tiếng Hàn)\n- Tip: Liên tưởng 한국 là tên nước và 어 là ngôn ngữ, kết hợp lại là \"tiếng của nước Hàn\"",
        "model": "gpt-4o-mini"
    },
    {
        "item": "공부하다",
        "analysis": "공부하다 (Học)\n- Nghĩa: học, nghiên cứu\n- Phân tích Hán-Hàn: 공부(công phu) = 공(công) là công sức + 부(phu) là đặt vào → đặt công sức vào việc học\n- Ví dụ: 저는 매일 한국어를 공부해요 (Tôi học tiếng Hàn mỗi ngày)\n- Tip: Liên tưởng với từ \"công phu\" trong tiếng Việt, học tập là quá trình đầu tư công sức.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "어려워요",
        "analysis": "어려워요 (Khó)\n- Nghĩa: khó, khó khăn\n- Ví dụ: 한국어는 어려워요 (Tiếng Hàn khó)\n- Tip: Từ này bắt nguồn từ tính từ 어렵다 (khó khăn), với đuôi 어요 biểu thị thì hiện tại.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "안녕하세요",
        "analysis": "안녕하세요 (Xin chào)\n- Nghĩa: Xin chào, chào hỏi thông thường\n- Phân tích: Từ 안녕 (bình an) + 하세요 (hình thức kính ngữ của động từ 하다)\n- Ví dụ: 안녕하세요, 저는 민수입니다. (Xin chào, tôi là Min-su)\n- Tip: Đây là cách chào hỏi phổ biến nhất trong tiếng Hàn, sử dụng trong hầu hết các tình huống gặp gỡ.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "감사합니다",
        "analysis": "감사합니다 (Cảm ơn)\n- Nghĩa: Cảm ơn (dạng kính ngữ)\n- Phân tích Hán-Hàn: 감사(cảm tạ) = 감(cảm) là cảm xúc + 사(tạ) là cảm ơn → cảm tạ, biết ơn\n- Ví dụ: 선물을 주셔서 감사합니다 (Cảm ơn vì đã tặng quà)\n- Tip: Đây là cách nói lịch sự, trang trọng hơn 고마워요.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "미안합니다",
        "analysis": "미안합니다 (Xin lỗi)\n- Nghĩa: Xin lỗi, tôi rất tiếc (dạng kính ngữ)\n- Phân tích Hán-Hàn: 미안(mĩ an) là từ Hán Hàn, có nghĩa là tiếc nuối, áy náy\n- Ví dụ: 늦어서 미안합니다 (Xin lỗi vì đến muộn)\n- Tip: Đây là cách xin lỗi trang trọng, lịch sự hơn 미안해요.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "책",
        "analysis": "책 (Sách)\n- Nghĩa: Sách, quyển sách\n- Phân tích Hán-Hàn: 책(sách) - đây là từ thuần Hàn, không phải từ Hán-Hàn\n- Ví dụ: 저는 한국어 책을 읽고 있어요 (Tôi đang đọc sách tiếng Hàn)\n- Tip: Một từ ngắn, dễ nhớ, phát âm giống với từ \"check\" trong tiếng Anh.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "학교",
        "analysis": "학교 (Trường học)\n- Nghĩa: Trường học\n- Phân tích Hán-Hàn: 학(học) là học + 교(giáo) là dạy dỗ → nơi dạy dỗ và học tập\n- Ví dụ: 저는 학교에 가요 (Tôi đi đến trường)\n- Tip: Liên tưởng đến từ \"học\" và \"giáo\" trong tiếng Việt, dễ nhớ vì cấu trúc và nghĩa tương tự.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "선생님",
        "analysis": "선생님 (Giáo viên)\n- Nghĩa: Giáo viên, thầy cô giáo\n- Phân tích Hán-Hàn: 선생(tiên sinh) = 선(tiên) là trước + 생(sinh) là người → người đi trước, người dẫn dắt; 님 là hậu tố kính ngữ\n- Ví dụ: 한국어 선생님은 친절해요 (Giáo viên tiếng Hàn rất tốt bụng)\n- Tip: 선생 có thể liên tưởng với \"tiên sinh\" (người đáng kính) trong tiếng Việt, thêm 님 để thể hiện sự tôn kính.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "친구",
        "analysis": "친구 (Bạn bè)\n- Nghĩa: Bạn, bạn bè\n- Phân tích Hán-Hàn: 친(thân) là thân thiết + 구(cữu) là người, bạn cũ → người thân thiết, bạn thân\n- Ví dụ: 그는 제 친구예요 (Anh ấy là bạn của tôi)\n- Tip: Liên tưởng từ \"thân cựu\" trong tiếng Việt, dễ nhớ vì có ý nghĩa tương tự.",
        "model": "gpt-4o-mini"
    }
]

# Sample GPT responses for grammar
SAMPLE_GRAMMAR_RESULTS = [
    {
        "item": "문법: (은|는) 것 같다\n예문: 한국어는 재미있는 것 같아요.",
        "analysis": "Cấu trúc ngữ pháp: (은/는) 것 같다\n\nNghĩa: Có vẻ như, dường như, có lẽ (diễn tả sự suy đoán dựa trên quan sát hoặc cảm nhận)\n\nCách sử dụng:\n- Động từ/tính từ + (으)ㄴ/는 것 같다\n- Với động từ ở thì hiện tại: 동사 + 는 것 같다\n- Với tính từ hoặc động từ ở thì quá khứ: 형용사/동사 + (으)ㄴ 것 같다\n\nPhân tích câu ví dụ:\n\"한국어는 재미있는 것 같아요\"\n- 한국어는: Tiếng Hàn (chủ ngữ với trợ từ 는)\n- 재미있는: thú vị (tính từ 재미있다 + 는)\n- 것 같아요: có vẻ như (kính ngữ)\n→ Nghĩa: Tiếng Hàn có vẻ thú vị.\n\nĐây là cấu trúc ngữ pháp quan trọng ở trình độ TOPIK trung cấp, dùng để diễn tả suy đoán, nhận xét mang tính chủ quan.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "문법: (으)?ㄹ 수 있다\n예문: 저는 한국어를 할 수 있어요.",
        "analysis": "Cấu trúc ngữ pháp: (으)ㄹ 수 있다\n\nNghĩa: Có thể, có khả năng (diễn tả khả năng hoặc năng lực làm việc gì đó)\n\nCách sử dụng:\n- Động từ + (으)ㄹ 수 있다\n- Nếu động từ kết thúc bằng nguyên âm hoặc ㄹ: 동사 + ㄹ 수 있다\n- Nếu động từ kết thúc bằng phụ âm (trừ ㄹ): 동사 + 을 수 있다\n\nPhân tích câu ví dụ:\n\"저는 한국어를 할 수 있어요\"\n- 저는: Tôi (chủ ngữ)\n- 한국어를: tiếng Hàn (tân ngữ)\n- 할 수 있어요: có thể nói/làm (동사 하다 + ㄹ 수 있다)\n→ Nghĩa: Tôi có thể nói tiếng Hàn.\n\nĐây là cấu trúc ngữ pháp cơ bản nhưng rất quan trọng ở trình độ TOPIK trung cấp, dùng để diễn tả khả năng làm việc gì đó.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "문법: (아|어|여) 보다\n예문: 한국 음식을 먹어 봤어요?",
        "analysis": "Cấu trúc ngữ pháp: (아/어/여) 보다\n\nNghĩa: Thử làm gì đó, đã từng trải nghiệm/làm một việc gì đó\n\nCách sử dụng:\n- Động từ + 아/어/여 보다\n  * Động từ có nguyên âm ㅏ, ㅗ: 동사 + 아 보다\n  * Động từ có các nguyên âm khác: 동사 + 어 보다\n  * Động từ 하다: 하 + 여 보다 = 해 보다\n\nPhân tích câu ví dụ:\n\"한국 음식을 먹어 봤어요?\"\n- 한국 음식을: món ăn Hàn Quốc (tân ngữ)\n- 먹어 봤어요?: đã từng ăn chưa? (먹다 + 어 보다 + 았어요 ở dạng câu hỏi)\n→ Nghĩa: Bạn đã từng ăn món Hàn Quốc chưa?\n\nĐây là cấu trúc ngữ pháp quan trọng ở trình độ TOPIK trung cấp, thường dùng để nói về kinh nghiệm đã trải qua hoặc đề nghị ai đó thử làm gì.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "문법: 기 때문에\n예문: 바빠서 못 갔어요.",
        "analysis": "Cấu trúc ngữ pháp: 기 때문에\n\nNghĩa: Bởi vì, vì, do...\n\nChú ý: Mặc dù câu ví dụ sử dụng dạng 아/어서 (바빠서), tôi sẽ phân tích cấu trúc 기 때문에 như được yêu cầu.\n\nCách sử dụng:\n- Động từ + 기 때문에\n- Thường dùng để giải thích lý do một cách chính thức, mạnh mẽ hơn so với 아/어서\n\nMột câu ví dụ với cấu trúc 기 때문에:\n\"바쁘기 때문에 못 갔어요\"\n- 바쁘기: từ động từ 바쁘다 (bận) + 기\n- 때문에: bởi vì\n- 못 갔어요: đã không thể đi\n→ Nghĩa: Vì bận nên đã không thể đi.\n\nCâu ví dụ gốc \"바빠서 못 갔어요\" sử dụng cấu trúc 아/어서, là cách diễn đạt nguyên nhân nhẹ nhàng hơn, thường dùng trong giao tiếp hàng ngày.\n\n기 때문에 là cấu trúc ngữ pháp quan trọng ở trình độ TOPIK trung cấp, dùng để diễn tả nguyên nhân một cách chính thức.",
        "model": "gpt-4o-mini"
    },
    {
        "item": "문법: (으)?면 안 되다\n예문: 여기서 담배를 피우면 안 돼요.",
        "analysis": "Cấu trúc ngữ pháp: (으)면 안 되다\n\nNghĩa: Không được làm gì, không nên làm gì (cấm đoán, ngăn cấm)\n\nCách sử dụng:\n- Động từ + (으)면 안 되다\n- Nếu động từ kết thúc bằng nguyên âm hoặc ㄹ: 동사 + 면 안 되다\n- Nếu động từ kết thúc bằng phụ âm (trừ ㄹ): 동사 + 으면 안 되다\n\nPhân tích câu ví dụ:\n\"여기서 담배를 피우면 안 돼요\"\n- 여기서: ở đây\n- 담배를: thuốc lá (tân ngữ)\n- 피우면: nếu hút (động từ 피우다 + 면)\n- 안 돼요: không được (안 되다 ở dạng kính ngữ)\n→ Nghĩa: Không được hút thuốc ở đây.\n\nĐây là cấu trúc ngữ pháp quan trọng ở trình độ TOPIK trung cấp, dùng để diễn tả điều cấm, quy định, hoặc lời khuyên không nên làm gì.",
        "model": "gpt-4o-mini"
    }
]


def generate_sample_data():
    """Generate sample data for testing."""
    return {
        'vocabulary_results': SAMPLE_VOCAB_RESULTS,
        'grammar_results': SAMPLE_GRAMMAR_RESULTS
    }


def main():
    """Export sample data to Excel for testing."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Generating sample data...")
    sample_data = generate_sample_data()
    
    output_path = "sample_output.xlsx"
    logger.info(f"Exporting sample data to: {output_path}")
    
    result_path = export_to_excel(sample_data, output_path)
    logger.info(f"Sample data exported to: {result_path}")


if __name__ == "__main__":
    main() 