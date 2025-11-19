from .expand_query import llm_expand_query
from .rag import retrieval_vdb
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
@chain
def get_docs(expanded_queries):
    all_docs=[]
    for q in expanded_queries:
        docs=retrieval_vdb(q,top_k=2,score_threshold=0.7)
        all_docs.extend(docs)
    # loai bo trung lap
    unique_contents = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in unique_contents:
            unique_contents.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs