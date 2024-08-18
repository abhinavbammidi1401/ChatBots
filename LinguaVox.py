from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Environment Variables
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Defining Translation Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a language translator. Translate the following text from {source_language} to {target_language}."),
        ("user", "Text: {text}")
    ]
)

# Streamlit UI
st.title("Language Translation using LangChain and Groq API")

# Adding more languages to the list
languages = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Italian", "Portuguese", "Russian", "Arabic", "Hindi", "Dutch", "Greek", "Swedish", "Turkish", "Vietnamese"]

source_language = st.selectbox("Select Source Language", languages)
target_language = st.selectbox("Select Target Language", languages)
inputText = st.text_input("Enter text to translate")

# Using Groq inference engine for translation
groqApi = ChatGroq(model="gemma-7b-It", temperature=0)
outputparser = StrOutputParser()
chainSec = prompt | groqApi | outputparser

# Execute Translation
if inputText:
    translation = chainSec.invoke({
        'source_language': source_language,
        'target_language': target_language,
        'text': inputText
    })

    # Display the translation result
    st.write(f"Translation ({source_language} to {target_language}):")
    st.markdown(f"**{translation}**")
