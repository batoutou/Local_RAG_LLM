import collections

from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class langchain :
    
    def __init__(self) -> None:
        
        self.text_template = """You're a helpful AI assistant tasked to answer the user's questions.
            You're friendly and you answer extensively with multiple sentences. 
            To help you, you can use the following context : {context}

            You prefer to use bullet points to summarize.

            Question: {question}
            """

        self.img_template = """Reformulate the following prompt that is used to generate images. Create a new proper prompt which only keeps the meaningful information and removing words that are not useful to the image's meaning.

            The prompt is : {question}
            please return only the new prompt and no surrounding text
            """
        
        self.querry_template = """Based on the following prompt, I just want to know whether the user is expecting a text answer or a generated image. 
            If it is suppose to be a text answer just return: text. If it is supposed to be a generated image just return:  img.
            The prompt is : {question}
            """
                        
        self.llm = self.get_llm()
        self.retriever = None

    def get_querry_type(self, question):

        prompt = self.get_prompt("querry")

        chain = prompt | self.llm | StrOutputParser()

        querry_type = chain.invoke({"question": question})  

        return querry_type
    
    def get_prompt(self, querry_type):
        
        if querry_type == "text":
            prompt = ChatPromptTemplate.from_template(self.text_template)
        elif querry_type == "img":
            prompt = ChatPromptTemplate.from_template(self.img_template)
        elif querry_type == "querry":
            prompt = ChatPromptTemplate.from_template(self.querry_template)
            
        return prompt
        
    def get_llm(self):
        llm = ChatOllama(model="llama3", temperature=0.1)
        
        return llm
    
    def get_chatbot_answer(self, question, context="", querry_type="text"):
        
        prompt = self.get_prompt(querry_type)   
        
        chain = prompt | self.llm | StrOutputParser()
        
        if querry_type == "text":
            answer = chain.invoke({"context": context, "question": question})    
        
        else:
            answer = chain.invoke({"question": question})       
                
        return answer