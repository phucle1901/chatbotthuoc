from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Đường dẫn đến file long-term memory
LONG_TERM_MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    'memory', 
    'long_term_memory.txt'
)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Bước 1: Kiểm tra xem query có chứa thông tin về đặc tính người dùng không
check_user_info_prompt = ChatPromptTemplate.from_messages([
    ('system', """Bạn là một AI chuyên phân tích câu hỏi của người dùng để trích xuất thông tin về đặc tính cá nhân.
    
Nhiệm vụ của bạn:
- Phân tích câu hỏi và xác định xem có chứa thông tin về đặc tính/đặc điểm của người dùng không
- Các thông tin cần chú ý: tuổi tác, giới tính, bệnh lý, dị ứng, tiền sử bệnh, thói quen sinh hoạt, sở thích, tình trạng sức khỏe, thuốc đang dùng, etc.

Trả về JSON với format:
{{
    "has_user_info": true/false,
    "user_info": "thông tin đặc tính người dùng (nếu có)",
    "reason": "lý do tại sao có/không có thông tin"
}}"""),
    ('human', 'Câu hỏi của người dùng: {user_query}')
])

# Bước 2: Kiểm tra xem có cần cập nhật vào long-term memory không
check_update_needed_prompt = ChatPromptTemplate.from_messages([
    ('system', """Bạn là một AI chuyên quản lý thông tin bộ nhớ dài hạn của người dùng.

Nhiệm vụ của bạn:
- So sánh thông tin mới với thông tin hiện tại trong bộ nhớ dài hạn
- Xác định xem có cần cập nhật không dựa trên các tiêu chí:
  + Thông tin mới và chưa có trong bộ nhớ
  + Thông tin bổ sung/chi tiết hơn thông tin cũ
  + Thông tin cập nhật/thay đổi so với trước
  + Thông tin quan trọng cho việc tư vấn thuốc

Trả về JSON với format:
{{
    "need_update": true/false,
    "update_action": "add/modify/skip",
    "reason": "lý do cần/không cần cập nhật",
    "updated_content": "nội dung cần thêm/cập nhật vào bộ nhớ (nếu cần)"
}}"""),
    ('human', """Thông tin người dùng mới: {user_info}
    
Bộ nhớ dài hạn hiện tại:
{current_memory}

Hãy quyết định có cần cập nhật không.""")
])

parser = JsonOutputParser()

@chain
def extract_user_info(user_query: str):
    """
    Bước 1: Trích xuất thông tin người dùng từ query
    """
    chain_check = check_user_info_prompt | llm | parser
    result = chain_check.invoke({"user_query": user_query})
    return result

@chain 
def check_update_needed(inputs: dict):
    """
    Bước 2: Kiểm tra xem có cần cập nhật long-term memory không
    """
    chain_check = check_update_needed_prompt | llm | parser
    result = chain_check.invoke({
        "user_info": inputs["user_info"],
        "current_memory": inputs["current_memory"]
    })
    return result

def read_long_term_memory():
    """Đọc nội dung hiện tại của long-term memory"""
    with open(LONG_TERM_MEMORY_PATH, 'r', encoding='utf-8') as f:
        return f.read().strip()


def update_long_term_memory(new_content: str, action: str = "add"):
    """
    Cập nhật long-term memory
    action: 'add' - thêm mới, 'modify' - sửa đổi toàn bộ
    """
    try:
        if action == "add":
            # Thêm thông tin mới vào cuối file
            current_content = read_long_term_memory()
            if current_content:
                updated_content = current_content + "\n" + new_content
            else:
                updated_content = new_content
        else:  # modify
            updated_content = new_content
            
        with open(LONG_TERM_MEMORY_PATH, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        return True
    except Exception as e:
        print(f"Lỗi khi cập nhật long-term memory: {e}")
        return False

@chain
def update_ltm_chatbot(user_query: str):
    """
    Chatbot chính để cập nhật long-term memory
    
    Args:
        user_query: Câu hỏi/phát biểu của người dùng
        
    Returns:
        dict: Kết quả quá trình xử lý
    """

    # Bước 1: Kiểm tra xem có thông tin người dùng không
    step1_result = extract_user_info.invoke(user_query)

    
    if not step1_result['has_user_info']:
        return {
            "status": "no_user_info",
            "message": "Không tìm thấy thông tin về đặc tính người dùng trong query",
            "detail": step1_result
        }
    
    # Bước 2: Kiểm tra xem có cần cập nhật không
    current_memory = read_long_term_memory()
    
    step2_result = check_update_needed.invoke({
        "user_info": step1_result['user_info'],
        "current_memory": current_memory
    })

    
    if not step2_result['need_update'] or step2_result['update_action'] == 'skip':
        return {
            "status": "no_update_needed",
            "message": "Không cần cập nhật long-term memory",
            "detail": step2_result
        }
    
    # Bước 3: Thực hiện cập nhật
    update_success = update_long_term_memory(
        new_content=step2_result['updated_content'],
        action=step2_result['update_action']
    )
    
    if update_success:

        return {
            "status": "updated",
            "message": "Đã cập nhật long-term memory thành công",
            "detail": {
                "step1": step1_result,
                "step2": step2_result,
                "updated_content": step2_result['updated_content']
            }
        }
    else:
        return {
            "status": "update_failed",
            "message": "Có lỗi khi cập nhật long-term memory",
            "detail": step2_result
        }
