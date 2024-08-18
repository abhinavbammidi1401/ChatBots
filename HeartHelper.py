from langchain_core.prompts import ChatPromptTemplate  # Prompt template
from langchain_core.output_parsers import StrOutputParser  # Default output parser whenever an LLM model gives any response
from langchain_groq import ChatGroq
import streamlit as st  # UI
import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith tracking (Observable)
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Using Groq inference engine
groqApi = ChatGroq(model="gemma-7b-It", temperature=0)
outputparser = StrOutputParser()

# Define the prompt template with a focus on relationship and dating advice
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Relationship and Dating Advisor. Your goal is to provide thoughtful, empathetic, and practical advice on relationship and dating topics. Please consider the user's feelings and offer respectful and helpful guidance."),
        ("user", "Question:{question}")
    ]
)
qa_chain = qa_prompt | groqApi | outputparser

# Streamlit UI
# Main content area
st.title("Relationship & Dating Advisor")

st.write("Welcome! I'm here to offer advice on relationships and dating. Feel free to ask me anything, whether it's about improving your love life, handling relationship issues, or planning a perfect date. I'm here to help!")

inputQuestion = st.text_input("Ask me anything about relationships or dating:")
if inputQuestion:
    response = qa_chain.invoke({'question': inputQuestion})
    st.write(response)
