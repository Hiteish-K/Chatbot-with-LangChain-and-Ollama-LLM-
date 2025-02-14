""" Chatbot with LangChain and Ollama LLM

A Streamlit-based chatbot powered by LangChain and Ollama LLM. This project demonstrates how to build an interactive chatbot that generates step-by-step answers to user questions. The app caches responses for faster performance and uses the `deepseek-r1` model for natural language understanding.

 > Features
- **Interactive Chat Interface**: Built with Streamlit for a user-friendly experience.
- **Response Caching**: Stores previously answered questions to improve response times.
- **LangChain Integration**: Uses LangChain's `ChatPromptTemplate` for prompt management.
- **Ollama LLM**: Leverages the `deepseek-r1` model for generating high-quality responses.

 > Technologies Used
- **Streamlit**: For building the web app.
- **LangChain**: For prompt templating and LLM integration.
- **Ollama LLM**: For natural language processing and response generation."""



#preinstall the following packages
#pip install --upgrade langchain langchain-community
#pip install --upgrade langchain_ollama
#pip install streamlit


from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

# Preload the model
if "model" not in st.session_state:
    st.session_state.model = OllamaLLM(model="deepseek-r1", temperature=0.7, max_tokens=100)

model = st.session_state.model

# Define prompt template
template = """Question:{question}
Answer=Generated Answer step by step"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit app
st.title(".com Chatbot")

# Cache responses
if "responses" not in st.session_state:
    st.session_state.responses = {}

question = st.text_input("Enter your question: ")

if question:
    if question in st.session_state.responses:
        response = st.session_state.responses[question]
    else:
        response = chain.invoke({"question": question})
        st.session_state.responses[question] = response
    st.write(response)