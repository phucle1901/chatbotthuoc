import os
def get_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Tính đường dẫn đến memory/long_term_memory.txt
    file_path = os.path.join(current_dir, '..', 'memory', 'long_term_memory.txt')
    with open(file_path,'r',encoding='utf-8') as f:
        content=f.read()
    return content