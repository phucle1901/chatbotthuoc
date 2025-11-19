import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from embedding import DrugEmbedding
from langchain_core.runnables import chain
import langchain
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
doc=DrugEmbedding()
def main():
    doc=DrugEmbedding()
    doc.load_index()
    llm=ChatOpenAI()
    prompt=ChatPromptTemplate.from_messages([
        ('system', """Bạn là một trợ lý y tế am hiểu về thuốc. Dựa trên thông tin về các loại thuốc được cung cấp, hãy trả lời các câu hỏi của người dùng một cách chính xác và ngắn gọn."""),
        ('human', """Dựa trên thông tin về các loại thuốc sau: {context}, hãy trả lời câu hỏi sau của người dùng: {query}""")
    ])
    @chain
    def f(query):
        docs=doc.vector_store.similarity_search(query,k=3)
        context="\n".join([d.page_content for d in docs])
        messages=prompt.format_messages(context=context,query=query)
        response=llm.invoke(messages)
        print(response.content)
    
    f.invoke("Tôi bị đau đầu, tôi nên dùng thuốc gì?")
    
if __name__ == "__main__":
    main()
