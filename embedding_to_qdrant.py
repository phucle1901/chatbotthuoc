

import os
from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from embedding import DrugEmbedding
import dotenv
from tqdm import tqdm
import time


def main(batch_size=10):
    """
    Migrate documents từ FAISS sang Qdrant theo batch
    
    Args:
        batch_size: Số lượng documents upload mỗi batch (mặc định: 10)
    """

    dotenv.load_dotenv()
    doc = DrugEmbedding()
    doc.load_index()
    
    # Trích xuất documents từ FAISS
    print("Đang trích xuất documents từ FAISS...")
    documents = []
    docstore = doc.vector_store.docstore
    index_to_docstore_id = doc.vector_store.index_to_docstore_id
    
    for idx in tqdm(range(len(index_to_docstore_id)), desc="Đọc documents"):
        doc_id = index_to_docstore_id[idx]
        document = docstore.search(doc_id)
        documents.append(document)
    
    print(f"Đã trích xuất {len(documents)} documents")
    print(f"Đang upload lên Qdrant (batch_size={batch_size})...")
    
    # Upload theo batch
    vector_store = None
    total_batches = (len(documents) - 1) // batch_size + 1
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Upload batches"):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        # Tạo Qdrant vector store cho batch đầu tiên
        if i == 0:
            vector_store = QdrantVectorStore.from_documents(
                documents=batch,
                embedding=doc.embeddings,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name="embedding_data",
                timeout=300
            )
        else:
            # Thêm batch tiếp theo vào collection
            vector_store.add_documents(batch)
    
    print(f"Đã migrate {len(documents)} documents sang Qdrant thành công!")
    print(f"Collection name: embedding_data")
    print(f"Tổng số batches: {total_batches}")

if __name__ == "__main__":
    main()
