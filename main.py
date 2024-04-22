import streamlit as st

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from rag_llm import rag_llm
from langChain import langchain

rag_llm = rag_llm()
langchain_llm = langchain()

custom_html = """
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
st.components.v1.html(custom_html)

# Sidebar content
st.sidebar.header("Tools")
st.sidebar.subheader("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload files to add to your knowledge data base")

st.title("Kickmaker AI bot !")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input(placeholder="Ask your question here !"):
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = langchain_llm.get_chatbot_answer(question)
            print("ANSWER : ", answer)
            st.write(answer)
        
    st.session_state.messages.append({"role": "assistant", "content": answer})