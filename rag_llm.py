import os
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.utils import filter_complex_metadata

class rag_llm:
    
    def __init__(self) -> None:
        
        # self.chroma_client = chromadb.Client()
        self.embeddings = OllamaEmbeddings(model="llama3")
        
        self.data_path = 'data'
        self.db_path = 'vec_db'
        
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)
        os.makedirs(self.data_path)
        
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        os.makedirs(self.db_path)
    
    def upload_data(self, uploaded_files):
        self.docs = []
        
        for file in uploaded_files:
            file_path = os.path.join(self.data_path, file.name)
            print("PATH :", file_path)
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
                
            loader = PyPDFLoader(file_path)
            self.docs.extend(loader.load())
                
    def make_vector_db(self):
        
        chunks = self.split_text(self.docs)
        self.retriever = self.save_to_chroma(chunks)
    
    def split_text(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        chunks = filter_complex_metadata(chunks)
        
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        
        # print(chunks[0])

        return chunks
    
    def save_to_chroma(self, chunks):
        # Create a new DB from the documents.
        db = Chroma.from_documents(
            chunks, self.embeddings, persist_directory = self.db_path
        )

        print(f"Saved {len(chunks)} chunks to {self.db_path}.")
        
        # retriever = db.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={
        #         "k": 3,
        #         "score_threshold": 0.5,
        #     },
        # )
        
        # return retriever