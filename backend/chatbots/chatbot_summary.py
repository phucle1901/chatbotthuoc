from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate   
prompt_template=ChatPromptTemplate.from_messages([
    ('system','Bạn là một chuyên gia tư vấn về thuốc và y tế. Hãy tóm tắt ngắn gọn các tài liệu dài sau đây để dễ dàng sử dụng trong việc trả lời câu hỏi của người dùng.'),
    ('human','Dưới đây là tài liệu dài cần tóm tắt: {long_text}')
    ])
llm=ChatOpenAI()
@chain
def summary(long_text):
    chatbot=prompt_template|llm
    summary_response=chatbot.invoke({
        'long_text':long_text
    })
    return summary_response.content