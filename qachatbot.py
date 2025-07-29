import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Langchain Groq Chatbot", page_icon=":robot_face:")

st.title("Langchain Groq Chatbot")
st.markdown("Learn langchain basic with Groq's fast inference!")


with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Enter your Groq API key",
        type="password",
        help="Get your API key from https://groq.com/console/api-keys/",
    )


    model_name = st.selectbox(
        "Model",
        ["groq-llama-3-70b","gemma2-9b-it"],
        index=0,
     )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


    ###LLM INITIALIZATION

@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    llm = ChatGroq(groq_api_key = api_key, model_name=model_name,
             temperature=0.7, streaming=True)   
   
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user","{questions}")
    ])    

    chain = prompt | llm | StrOutputParser()
    return chain


####### CHAIN EXECUTION
chain = get_chain(api_key, model_name)

if not chain:
    st.error("Please enter your Groq API key to continue.")
    st.markdown("Get your API key from [Groq Console](https://groq.com/console/api-keys/).")
else:

### DISPLAYING CHAT
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write (message["content"])

    if question:= st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role":"user","content": "question"})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream response from Groq
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")    


st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

st.markdown("---")
st.markdown("Built with LangChain & Groq | Experience the speed! ⚡")

        
