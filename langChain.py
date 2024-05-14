import collections

from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class langchain :
    def __init__(self) -> None:
        # different LangChain template for the various cases 
        # template to answer to a question about anything including things inside a PDF file
        self.text_template = """You're a helpful AI assistant tasked to answer the user's questions.
            You're friendly and you answer extensively with multiple sentences. 
            To help you, you can use the following context : {context}

            You prefer to use bullet points to summarize.

            Question: {question}
            """

        # template to generate an image based on a prompt
        self.img_template = """Reformulate the following prompt that is used to generate images. Create a new proper prompt which only keeps the meaningful information and removing words that are not useful to the image's meaning.

            The prompt is : {question}
            please return only the new prompt and no surrounding text
            """
        
        # template to create a resume of a PDF file which will use only the context from this file 
        self.resume_template = """You're a helpful AI assistant tasked to resume PDF files.
            You're friendly and you answer extensively with multiple sentences. 
            To answer, please use only the following context : {context}
            
            Question: {question}
            """
        
        # template to determine the query type between Text, Image and Resume
        self.query_template = """Based on the following prompt, I just want to know whether the user is expecting a text answer or a generated image or a resume of a given pdf file. 
            If it is suppose to be a text answer just return: text. If it is supposed to be a generated image just return:  img. If it is supposed to be a resume just return:  resume.
            The prompt is : {question}
            If you are not sure : say that you don't understand the query
            """
                        
        self.llm = self.get_llm()
        self.retriever = None

    # get the query type 
    def get_query_type(self, question):

        # create a prompt based on the query template
        prompt = self.get_prompt("query")
        # create a LangChain chain with the prompt and the llm
        chain = prompt | self.llm | StrOutputParser()
        # recover the type
        query_type = chain.invoke({"question": question})  

        return query_type
    
    # get the prompt  
    def get_prompt(self, query_type):
        # pick the adequate query type and return the corresponding prompt 
        if query_type == "text":
            prompt = ChatPromptTemplate.from_template(self.text_template)
        elif query_type == "img":
            prompt = ChatPromptTemplate.from_template(self.img_template)
        elif query_type == "resume":
            prompt = ChatPromptTemplate.from_template(self.resume_template)
        elif query_type == "query":
            prompt = ChatPromptTemplate.from_template(self.query_template)
            
        return prompt
    
    # get the LLM instance    
    def get_llm(self):
        # create an instance of the LLM model, in our case the LLAMA3 from Meta
        llm = ChatOllama(model="llama3", temperature=0.1)
        
        return llm
    
    def get_chatbot_answer(self, question, context="", query_type="text"):
        # get the correct prompt
        prompt = self.get_prompt(query_type)   
        # create the chain
        chain = prompt | self.llm | StrOutputParser()
        # if we want an answer or a resume
        if query_type == "text" or "resume":
            # get the answer back by passing context and the question
            answer = chain.invoke({"context": context, "question": question})    
        
        else:
            # get the answer back by passing only the question
            answer = chain.invoke({"question": question})       
                
        return answer