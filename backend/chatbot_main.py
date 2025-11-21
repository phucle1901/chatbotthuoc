from get_doc.get_docs import get_docs
from get_doc.get_long_term import get_data
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate   
from get_doc.expand_query import llm_expand_query
from chatbots.chatbot_summary import summary
from tracing import tracer
from chatbots.chatbot_update_ltm import update_ltm_chatbot
prompt_template=ChatPromptTemplate.from_messages([
    ('system','Bạn là một chuyên gia tư vấn về thuốc và y tế. Sử dụng thông tin từ bộ nhớ dài hạn và các tài liệu liên quan để trả lời câu hỏi của người dùng một cách chính xác và chi tiết.'),
    ('human',"""Câu hỏi của người dùng: {user_query}
     Dữ liệu bộ nhớ dài hạn: {long_term_memory} 
     Tài liệu tham khảo: {reference_docs}
     Trong trường hợp nếu cảm thấy thông tin từ bộ nhớ dài hạn và tài liệu tham khảo không đủ để trả lời câu hỏi , bạn hãy dựa vào kiến thức của bản thân để tra lời câu hỏi(trong câu trả lời cho người dùng thì không cần nói tài liệu tham khảo không có thông tin )""")
    ])
llm=ChatOpenAI(model="gpt-4")
@chain
def chatbot_response(user_query):
    with tracer.start_as_current_span("chatbot_response"):
        #cap nhat longterm neu can:
        with tracer.start_as_current_span("update_long_term_memory"):
            update_ltm_chatbot.invoke(user_query)
        # lay data longterm
        with tracer.start_as_current_span("get_long_term"):
            data_longterm = get_data()

        # mo rong query
        with tracer.start_as_current_span("expand_query"):
            expand_query = llm_expand_query.invoke(user_query)

        # lay docs
        with tracer.start_as_current_span("get_docs"):
            docs = get_docs.invoke(expand_query)

        # kiem tra do dai + summarize khi cần
        combined_doc_content = ""
        with tracer.start_as_current_span("summarize"):
            for doc in docs:
                combined_doc_content += doc.page_content + "\n"
                    if len(combined_doc_content) > 10000:
                        combined_doc_content = summary.invoke(combined_doc_content)

        # tra loi (LLM invocation)
        with tracer.start_as_current_span("llm_invoke"):
            chatbot = prompt_template | llm
            chatbot_response = chatbot.invoke({
                'user_query': user_query,
                'long_term_memory': data_longterm,
                'reference_docs': combined_doc_content
            })

        return chatbot_response.content