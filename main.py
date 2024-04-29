import os 
import sys
import shutil

import streamlit as st

from rag_llm import rag_llm
from langChain import langchain
from img_gen import img_gen

query_llm = rag_llm()
langchain_llm = langchain()
image_gen = img_gen()

custom_html = """
        <div class="banner">
            <img src="https://www.les-soudes.com/app/uploads/2021/08/logo_kickmaker_CMJN_black-1-Simon-Fressy.png" alt="Banner Image">
        </div>
        <style>
            .banner {
                width: 100%;
                height: 100%;
                overflow: hidden;
            }
            .banner img {
                width: 100%;
                object-fit: fill;
            }
        </style>
        """

# Display the custom HTML
st.set_page_config(layout="wide")
st.components.v1.html(custom_html)

# Sidebar content
st.sidebar.header("Tools")
st.sidebar.subheader("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload files to add to your knowledge data base", type=['pdf'], accept_multiple_files=True)

st.title("Kickmaker AI bot !")

if uploaded_file:
    query_llm.upload_data(uploaded_file)
    
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input(placeholder="Ask your question here !"):
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    querry_type = langchain_llm.get_querry_type(question)
    # if "generate" and "image" in question:
    if querry_type == "img":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):  
                prompt = langchain_llm.get_chatbot_answer(question, querry_type="img")
                image = image_gen.generate_img(question)  
                st.image(image, caption=question, use_column_width=True)
    
    else:   
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = query_llm.search_chroma(question)
                answer = langchain_llm.get_chatbot_answer(question, context, querry_type="text")
                st.write(answer)
            
        st.session_state.messages.append({"role": "assistant", "content": answer})