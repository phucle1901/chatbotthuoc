"""
BÀI TẬP: TẠO HỆ THỐNG EMBEDDING VÀ TÌM KIẾM THUỐC BẰNG FAISS

Mục tiêu: Xây dựng một class để:
1. Đọc dữ liệu thuốc từ file JSON
2. Kết hợp các thuộc tính của mỗi loại thuốc thành 1 đoạn text
3. Tạo embeddings bằng OpenAI
4. Lưu vào FAISS index để tìm kiếm

Kiến thức cần có:
- Python cơ bản (class, list, dict, vòng lặp)
- Đọc/ghi file JSON
- Sử dụng thư viện pathlib để xử lý đường dẫn
"""

import json
from pathlib import Path
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # ✨ Thay OpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import pickle
import dotenv 
dotenv.load_dotenv()


# ========== BƯỚC 2: TẠO CLASS DrugEmbedding ==========
class DrugEmbedding:
    """
    Class chính để xử lý embedding và tìm kiếm thuốc
    """
    
    def __init__(self, data_path: str = "./drugs-data-main/data"):
        """
        Hàm khởi tạo - Chạy đầu tiên khi tạo object DrugEmbedding
        
        Args:
            data_path: Đường dẫn đến thư mục chứa dữ liệu thuốc
        
        Nhiệm vụ:
        1. Lưu đường dẫn dữ liệu
        2. Tạo đường dẫn đến thư mục details (chứa các file JSON thuốc)
        3. Khởi tạo OpenAIEmbeddings (cần OPENAI_API_KEY trong biến môi trường)
        4. Khởi tạo các biến lưu trữ: vector_store và drugs_data
        """

        self.data_path = Path(data_path)        
        self.details_path = self.data_path / "details"
        self.embeddings=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"  
        )
        self.vector_store = None  # Sẽ lưu FAISS index
        self.drugs_data = []      # Sẽ lưu danh sách thuốc đã đọc
    
    
    def load_drug_data(self) -> List[Dict]:
        """
        Đọc tất cả file JSON thuốc từ thư mục details
        
        Returns:
            List[Dict]: Danh sách các dictionary, mỗi dict chứa thông tin 1 loại thuốc
        
        Thuật toán:
        1. Tạo list rỗng để chứa dữ liệu thuốc
        2. Kiểm tra xem thư mục details có tồn tại không
        3. Duyệt qua tất cả thư mục con (danh mục thuốc) trong details
        4. Với mỗi danh mục, đọc tất cả file .json
        5. Load nội dung JSON và thêm vào list
        6. Trả về list đã đọc
        """
        # TODO: Tự hoàn thiện hàm này
        # Gợi ý các bước:
        
        # Bước 1: Tạo list rỗng
        drugs = []
        for category_dir in self.details_path.iterdir():
            category_name=category_dir.name
            print(f"Đang xử lý: {category_name}")
            for json_file in category_dir.glob("*.json"):
                with open(json_file,'r',encoding='utf-8') as f:
                    drug_data=json.load(f)
                    drug_data['category']=category_name
                    drug_data['file_name']=json_file.stem
                    drugs.append(drug_data)
                    
        self.drugs_data=drugs
        return drugs

    
    
    def combine_drug_attributes(self, drug: Dict) -> str:
        """
        Kết hợp tất cả thuộc tính của 1 loại thuốc thành 1 chuỗi text
        
        Args:
            drug: Dictionary chứa thông tin thuốc (đọc từ JSON)
        
        Returns:
            str: Chuỗi text đã kết hợp tất cả thuộc tính
        
        Ví dụ input:
        {
            "category": "Cơ-xương-khớp",
            "file_name": "thuoc-giam-dau",
            "describe": "Thuốc giảm đau...",
            "ingredient": "Paracetamol 500mg",
            ...
        }
        
        Ví dụ output:
        '''
        Danh mục:
        Cơ-xương-khớp
        
        Tên file:
        thuoc-giam-dau
        
        Mô tả:
        Thuốc giảm đau...
        
        Thành phần:
        Paracetamol 500mg
        ...
        '''
        """

        fields = [
            ('Danh mục', drug.get('category', '')),
            ('Tên thuốc', drug.get('file_name', '')),
            ('Mô tả', drug.get('describe', '')),
            ('Thành phần', drug.get('ingredient', '')),
            ('Công dụng', drug.get('usage', '')),
            ('Liều dùng', drug.get('dosage', '')),
            ('Tác dụng phụ', drug.get('adverse_effect', '')),
            ('Lưu ý', drug.get('careful', '')),
            ('Bảo quản', drug.get('preservation', ''))
        ]
        combined_text=""
        for field_name,field_data in fields:
            combined_text += f"\n{field_name}:\n{field_data}\n"
        combined_text=combined_text.strip()
        return combined_text

    
    def create_documents(self) -> List[Document]:
        """
        Tạo danh sách Document từ dữ liệu thuốc
        Document là định dạng mà LangChain yêu cầu để tạo embeddings
        
        Returns:
            List[Document]: Danh sách các Document
        
        Cấu trúc Document:
        - page_content: Nội dung text (từ combine_drug_attributes)
        - metadata: Thông tin bổ sung (id, category, file_name, source)
        """

        if not self.drugs_data:
            self.load_drug_data()

        documents=[]
        for index,drug in enumerate(self.drugs_data):
            combined_text=self.combine_drug_attributes(drug)
            metadata={
                'id':index,
                'category':drug['category'],
                'file_name':drug['file_name'],
            }
            doc = Document(page_content=combined_text, metadata=metadata)
            documents.append(doc)
        return documents

    
    def create_embeddings_and_index(self, save_path: str = "./faiss_index",batch_size=10):
        """
        Tạo embeddings cho tất cả documents và lưu vào FAISS index
        
        Args:
            save_path: Đường dẫn để lưu FAISS index
        
        Returns:
            FAISS vector store
        
        Quy trình:
        1. Tạo documents từ dữ liệu thuốc
        2. Sử dụng OpenAI để tạo embeddings cho mỗi document
        3. FAISS sẽ tự động tạo index từ embeddings
        4. Lưu index vào disk
        """
        print("📚 Đang tạo documents...")
        documents = self.create_documents()
        print(f"✅ Đã tạo {len(documents)} documents")
        print(f"🔄 Đang tạo embeddings (batch_size={batch_size})...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"   Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}: {len(batch)} documents")
        
            # Tạo FAISS index cho batch đầu tiên
            if i == 0:
                self.vector_store = FAISS.from_documents(
                    documents=batch,
                    embedding=self.embeddings
                )
            else:
                # Merge batch tiếp theo vào vector_store
                batch_store = FAISS.from_documents(
                    documents=batch,
                    embedding=self.embeddings
                )
                self.vector_store.merge_from(batch_store)
    
        print("✅ Đã tạo xong embeddings!")
        print("💾 Đang lưu index...")
        self.save_index(save_path)
        print(f"✅ Đã lưu index vào {save_path}")
    

        return self.vector_store
        

    
    def save_index(self, save_path: str = "./faiss_index"):
        """
        Lưu FAISS index và dữ liệu thuốc vào disk
        
        Args:
            save_path: Đường dẫn để lưu
        """

        
        Path(save_path).mkdir(parents=True,exist_ok=True)
        self.vector_store.save_local(save_path)
        path=Path(save_path)/"drugs_data.pkl"
        with open(path,'wb') as f:
            pickle.dump(self.drugs_data,f)
            
    
    
    def load_index(self, load_path: str = "./faiss_index"):
        """
        Load FAISS index đã lưu từ disk
        
        Args:
            load_path: Đường dẫn chứa index
        """

        self.vector_store=FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        with open(Path(load_path) / "drugs_data.pkl", "rb") as f:
            self.drugs_data = pickle.load(f)
        
    
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Tìm kiếm thuốc dựa trên câu hỏi
        
        Args:
            query: Câu hỏi (VD: "thuốc trị đau đầu")
            k: Số lượng kết quả trả về
        
        Returns:
            List các kết quả tìm kiếm (có điểm tương đồng)
        """

        result=self.vector_store.similarity_search_with_score(query,k=k)
        format_results=[]
        for doc,score in result:
            format_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
        return format_results


# ========== BƯỚC 3: HÀM MAIN ĐỂ CHẠY CHƯƠNG TRÌNH ==========
def main():
    """
    Hàm main - điểm bắt đầu của chương trình
    
    Quy trình:
    1. Khởi tạo DrugEmbedding
    2. Tạo embeddings và FAISS index
    3. Test tìm kiếm
    """
    # TODO: Tự hoàn thiện
    # Gợi ý:
    
    # Bước 1: Tạo object DrugEmbedding
    drug_embedding = DrugEmbedding(data_path="./drugs-data-main/data")
    
    # Bước 2: Tạo embeddings và index
    drug_embedding.create_embeddings_and_index(save_path="./faiss_index")
    
    # Bước 3: Test tìm kiếm
    results = drug_embedding.search("thuốc trị đau xương khớp", k=3)
    for i, result in enumerate(results, 1):
        print(f"Kết quả {i}: {result['metadata']['file_name']}")
    
    


# Chạy chương trình
if __name__ == "__main__":
    main()

