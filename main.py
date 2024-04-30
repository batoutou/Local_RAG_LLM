import streamlit as st

from rag_llm import rag_llm
from langChain import langchain
from img_gen import img_gen

# create instances for each process
query_llm = rag_llm()
langchain_llm = langchain()
image_gen = img_gen()

# custom HTML code for visual 
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
# create a file uploader module 
uploaded_file = st.sidebar.file_uploader("Upload files to add to your knowledge data base", type=['pdf'], accept_multiple_files=True)

# print the main title
st.title("Kickmaker AI bot !")

# check if a new pdf file was uploaded
if uploaded_file:
    # run the new file through the embedding DB creator
    query_llm.upload_data(uploaded_file)

# If the program is starting and there is no messages at all
if "messages" not in st.session_state:
    # print a welcoming message to engage the discussion"
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# check the message buffer and display the new ones
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# check if the user has ask something 
if question := st.chat_input(placeholder="Ask your question here !"):
    # send the question to the message buffer
    st.session_state["messages"].append({"role": "user", "content": question})
    # display the message
    st.chat_message("user").markdown(question)

    # check whether the user wants a text answer or a generated image
    querry_type = langchain_llm.get_querry_type(question)
    
    # if he wants an image
    if querry_type == "img":
        # display a spinning circle will the program runs
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):  
                # create an adequate LangChain prompt to generate images
                prompt = langchain_llm.get_chatbot_answer(question, querry_type="img")
                # call the Stable Diffusion model to generate the image
                image = image_gen.generate_img(question)  
                # send the image to the message buffer
                st.image(image, caption=question, use_column_width=True)
        
        st.session_state.messages.append({"role": "assistant", "content": image})
    
    # if he wants a text
    else:   
        # display a spinning circle will the program runs
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # querry the pdf DB for useful data
                context = query_llm.search_chroma(question)
                # get a proper answer based on this useful data and the model's knowledge 
                answer = langchain_llm.get_chatbot_answer(question, context, querry_type="text")
                # send the text to the message buffer
                st.write(answer)
            
        st.session_state.messages.append({"role": "assistant", "content": answer})