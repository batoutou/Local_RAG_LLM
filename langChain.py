import collections

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class langchain :
    
    def __init__(self) -> None:
        
        self.template = """You're a helpful AI assistant tasked to answer the user's questions.
            You're friendly and you answer extensively with multiple sentences. 
            You prefer to use bullet points to summarize.

            Question: {question}

            Helpful Answer:"""
                        
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
    
    def get_prompt(self):
        prompt = ChatPromptTemplate.from_template(self.template)
        return prompt
    
    def get_embeddings(self) -> OllamaEmbeddings:
        embeddings = OllamaEmbeddings(model="llama3")
        return embeddings
    
    def get_llm(self):
        llm = ChatOllama(model="llama3", temperature=0.2)
        
        return llm
    
    def get_chatbot_answer(self, question):
        
        prompt = self.get_prompt()        
        chain = prompt | self.llm | StrOutputParser()
                
        answer = chain.invoke({"question": question})                
                
        return answer