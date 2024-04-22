import streamlit as st
from threading import Thread

from langChain import langchain

class interface :
    
    def __init__(self) -> None:
        self.chain = langchain()
        
        self.custom_html = """
        <div class="banner">
            <img src="https://www.les-soudes.com/app/uploads/2021/08/logo_kickmaker_CMJN_black-1-Simon-Fressy.png" alt="Banner Image">
        </div>
        <style>
            .banner {
                width: 100%;
                height: 200px;
                overflow: hidden;
            }
            .banner img {
                width: 100%;
                object-fit: cover;
            }
        </style>
        """
        # Display the custom HTML
        st.set_page_config(layout="wide")
        st.components.v1.html(self.custom_html)
        
        # Sidebar content
        st.sidebar.header("Tools")
        st.sidebar.subheader("Upload Files")
        self.uploaded_file = st.sidebar.file_uploader("Upload files to add to your knowledge data base")

        st.title("Kickmaker AI bot !")
        
        thread = Thread(target = self.update_interface)
        thread.start()
    
    def update_interface(self):
        
        while True:
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "Comment puis-je vous aider ?"}]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
    
    def input_query(self):
        
        # user input question
        if question := st.text_input("Enter your request", placeholder="I wish to know if Santa is real ?"):
            
            # display user question
            st.session_state.messages.append({"role": "user", "content": question})
            # st.chat_message("user").write(question)
            
            # Generate the answer
            answer = self.chain.get_chatbot_answer(question)
            
            # st.chat_message("assistant").write(answer)
            
            # print("answer is :", answer)
                        
            # display chatbot answer
            st.session_state.messages.append({"role": "assistant", "content": answer})
            