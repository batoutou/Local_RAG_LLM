from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

from langchain_community.embeddings import OllamaEmbeddings

class rag_llm:
    
    def __init__(self) -> None:
        
        # self.chroma_client = chromadb.Client()
        self.embeddings = OllamaEmbeddings(model="llama3")
    
    def upload_data(self, path):
        pass
    
    def generate_data_store(self):
        
        documents = self.load_documents()
        chunks = self.split_text(documents)
        self.save_to_chroma(chunks)
    
    def load_documents(self, data_path):
        loader = DirectoryLoader(data_path, glob="*.md")
        documents = loader.load()
        return documents


    def split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

        return chunks
    
    def save_to_chroma(self, chunks:list[Document], db_path):
        # Clear out the database first.
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        # Create a new DB from the documents.
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=db_path
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {db_path}.")