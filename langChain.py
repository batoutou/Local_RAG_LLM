import collections

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
                        
        self.llm = self.get_llm()
        self.retriever = None
    
    def get_prompt(self):
        prompt = ChatPromptTemplate.from_template(self.template)
        return prompt
        
    def get_llm(self):
        llm = ChatOllama(model="llama3", temperature=0.2)
        
        return llm
    
    def get_chatbot_answer(self, question):
        
        prompt = self.get_prompt()   
        if self.retriever: 
            chain = ({"context": self.retriever} | prompt | self.llm | StrOutputParser())
        else: 
            chain = prompt | self.llm | StrOutputParser()
                
        answer = chain.invoke({"question": question})                
                
        return answer