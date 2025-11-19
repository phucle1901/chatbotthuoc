import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables  import chain
llm=ChatOpenAI()
prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'Bạn là một chuyên gia tư vấn về thuốc và y tế.'),
    ('human', """Câu hỏi của người dùng: {query}

Nhiệm vụ: Hãy mở rộng câu hỏi trên thành 4 câu hỏi để tăng cường ngữ cảnh và hiểu rõ hơn ý định của người dùng.

Yêu cầu:
- Câu 1: Giữ nguyên câu hỏi gốc
- Câu 2-4: Tạo 3 câu hỏi mở rộng liên quan đến thuốc, công dụng, liều lượng, tác dụng phụ, hoặc cách sử dụng

Định dạng đầu ra (mỗi câu trên một dòng):
1. [Câu hỏi gốc]
2. [Câu hỏi mở rộng về thành phần/công dụng]
3. [Câu hỏi mở rộng về liều lượng/cách dùng]
4. [Câu hỏi mở rộng về tác dụng phụ/lưu ý]

Ví dụ:
1. Paracetamol dùng để làm gì?
2. Paracetamol có thành phần chính là gì và công dụng chính là gì?
3. Liều lượng và cách sử dụng Paracetamol như thế nào?
4. Paracetamol có tác dụng phụ gì cần lưu ý?""")
])
def split_query(q):
    queries=[query.strip() for query in q.content.split('\n')]
    return queries
expand_query=prompt_template|llm|split_query

@chain
def llm_expand_query(query):
    return expand_query.invoke(query)