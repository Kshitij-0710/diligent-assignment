import streamlit as st
import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM, OllamaEmbeddings 
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
if not os.getenv("PINECONE_API_KEY"):
    st.error("Error: PINECONE_API_KEY not found in .env file!")
    st.stop()

INDEX_NAME = "jarvis-demo"

st.set_page_config(page_title="Jarvis", page_icon="🤖")
st.title("🤖 Jarvis")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

llm = OllamaLLM(model="llama3.2:1b") 
embeddings = OllamaEmbeddings(model="llama3.2:1b")


for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Type here..."):
    
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        try:
            vectorstore = PineconeVectorStore.from_existing_index(INDEX_NAME, embeddings)
            
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 3}
            )
            
            current_history = st.session_state.chat_history

            template = """You are a helpful assistant.
            Use the following pieces of retrieved context to answer the question.
            If the context isn't relevant to the question, ignore it and just chat normally.
            
            Context: {context}
            
            Question: {question}
            """
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", template),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])
            
            chain = (
                {
                    "context": retriever, 
                    "question": RunnablePassthrough(), 
                    "history": lambda x: current_history
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            response = chain.invoke(prompt)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
            
        except Exception as e:
            st.warning(f"Retrieval skipped or failed: {e}")