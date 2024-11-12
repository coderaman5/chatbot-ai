# app.py

import streamlit as st
from langchain.llms import LlamaCpp

# Initialize the LlamaCpp model
llm = LlamaCpp(
    model_path="phi-3-mini-4k-instruct-fp16.gguf",  # Update with your model path
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=200,
    seed=42,
    verbose=False,
)

# Streamlit app configuration
st.set_page_config(page_title="Chat with LlamaCpp", layout="centered")

# Header
st.title("Chat with LlamaCpp Model")
st.write("Type your question below and get a response from the AI. Type 'exit' to end the conversation.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input section
user_input = st.text_input("You:", key="input")

# If user input exists
if user_input:
    # Add user input to chat history
    st.session_state.chat_history.append(("User", user_input))

    # Exit condition
    if user_input.lower() == "exit":
        st.write("Chat ended. Refresh to start a new conversation.")
    else:
        # Construct the prompt for LlamaCpp model
        prompt = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"
        
        # Generate response from the model
        response = llm(
            prompt,
            max_tokens=256,
            stop="<|end|>",
            echo=False,
        )
        
        # Add model's response to chat history
        st.session_state.chat_history.append(("Assistant", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "User":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Assistant:** {message}")
