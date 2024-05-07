import os
import shutil

import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQAWithSourcesChain

class rag_llm:
    def __init__(self) -> None:
        # get vector embeddings model
        self.embeddings = OllamaEmbeddings(model="llama3")
        
        # path to data and vector DB
        self.data_path = 'data'
        self.db_path = 'vec_db'

        # create the vector database
        self.get_chroma()
    
    # upload PDF file to the system                  
    def upload_data(self, uploaded_files):
        self.docs = []
        # iterate over uploaded files
        for file in uploaded_files:
            # if the document isn't already present, save it to a tmp folder
            if not os.path.isfile(os.path.join("data", file.name)):
                file_path = os.path.join(self.data_path, file.name)
                with open(file_path, 'wb') as f:
                    f.write(file.getvalue())
                # use a PDF loader library to read the file
                loader = PyPDFLoader(file_path)
                self.docs.extend(loader.load())
        
        # if new documents added
        if self.docs:
            # save them to the vector database
            self.add_chunks()

    # read PDF files and store them into the vector database           
    def add_chunks(self):
        # create chunks of the file
        chunks = self.split_text(self.docs)
        # save the chunks to the database as
        self.save_to_chroma(chunks)
    
    # split the pdf in smaller chunks and filter metadata
    def split_text(self, documents):
        # instance for a text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,        
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        # make the chunks
        chunks = text_splitter.split_documents(documents)
        # filter the chunks
        chunks = filter_complex_metadata(chunks)
        
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        
        return chunks
    
    # save the documents as chunks to the vector database
    def save_to_chroma(self, chunks):
        # Create a new DB from the documents.
        self.db.add_documents(chunks)
        # persist the DB so that it remains available
        self.db.persist()

        print(f"Saved {len(chunks)} chunks to {self.db_path}.")
        
    # create or get an exitsing vector DB
    def get_chroma(self):
        # create folder to save the DB
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        # if the DB already exist
        if os.path.isfile("vec_db/chroma.sqlite3"):
            # get the DB
            self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        
        else:
            # create a DB client and a collection 
            client = chromadb.PersistentClient(path=self.db_path)
            collection = client.get_or_create_collection("Documents")
            # create the DB and persist it
            self.db = Chroma(persist_directory=self.db_path, client=client, collection_name="Documents", embedding_function=self.embeddings)

    # search within the DB for relevant data 
    def search_chroma(self, question, query_type):
        # if we ask for a text answer
        if query_type == "text":
            # Search the DB for 5 closest chunks of data
            results = self.db.similarity_search_with_score(question, k=5)
            context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # if we want a PDF resume
        elif query_type == "resume":
            # get all the chunks
            results = self.db.get()
            context = "\n\n---\n\n".join(results["documents"])    

        return context
    
    # remove the database 
    def remove_chroma(self):
        # if the data folder exist
        if os.path.isdir(self.data_path):
            # remove it
            shutil.rmtree(self.data_path)
        # delete the vector DB collection
        self.db.delete_collection()
        self.docs = None