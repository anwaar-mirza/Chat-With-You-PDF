from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import streamlit as st
import tempfile
import string
import random
import os

load_dotenv()

os.environ['HF_TOKEN'] = st.secrets['HF_TOKEN']


class RunnableChatBotWithHistory:
    
    def __init__(self, groq_api, file_to_split):
        self.groq_api = groq_api
        self.file_to_split = file_to_split

    def load_and_split_documents(self):
        loader = PyPDFLoader(self.file_to_split)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=250)
        split_documents = splitter.split_documents(documents)
        return split_documents
    
    def create_llm(self):
        return ChatGroq(api_key=self.groq_api, model="gemma2-9b-it")
    
    def create_embeddings(self):
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def create_retriever(self):
        vector_store = FAISS.from_documents(self.load_and_split_documents(), self.create_embeddings())
        return vector_store.as_retriever(search_kwargs={"k": 5})
    
    def create_contextual_prompt(self):
        contextualize_q_system_prompt = """
            <prompt>
                <role>
                    <name>QuestionContextualizer</name>
                    <description>
                        A system that transforms user questions into standalone questions using the provided chat history.
                    </description>
                </role>

                <goal>
                    <primary>
                        To rewrite user questions that rely on previous conversation context into fully self-contained, standalone questions.
                    </primary>
                    <secondary>
                        If the question is already standalone, return it as-is without any changes.
                    </secondary>
                </goal>

                <instructions>
                    <step>1. Receive the full chat history and the latest user question.</step>
                    <step>2. Analyze whether the latest question references earlier context (e.g., pronouns like "he", "that", "it").</step>
                    <step>3. Reformulate the question to include necessary details from the chat history, making it understandable without the history.</step>
                    <step>4. Do NOT answer the question—only rewrite it if necessary.</step>
                    <step>5. If no reformulation is needed, return the original question unchanged.</step>
                </instructions>
            </prompt>
            """
        contextual_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        return contextual_prompt
    
    def create_actual_system_prompt(self):
        actual_system_prompt = """
            <prompt>
                <role>
                    <name>AnswerAssistant</name>
                    <description>
                        An AI assistant designed to answer user questions using retrieved contextual information.
                    </description>
                </role>

                <goal>
                    <primary>
                        To provide accurate and concise answers to user questions using only the supplied context.
                    </primary>
                    <secondary>
                        To admit when the answer is unknown rather than guessing or fabricating a response.
                    </secondary>
                </goal>

                <instructions>
                    <step>1. Read the retrieved context provided.</step>
                    <step>2. Use only the given context to generate an answer to the user’s question.</step>
                    <step>3. If the context does not contain enough information, respond with "I don't know."</step>
                    <step>4. Limit the response to a maximum of three sentences.</step>
                    <step>5. Keep the answer clear and concise.</step>
                </instructions>
                <Context>{context}</Context>
            </prompt>

            """
        
        system_prompt = ChatPromptTemplate([
            ("system", actual_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        return system_prompt
    
    def create_history_retriever(self):
        history_aware_retriever = create_history_aware_retriever(self.create_llm(), self.create_retriever(), self.create_contextual_prompt())
        return history_aware_retriever
    
    def create_retrieval_chain(self):
        doc_chain = create_stuff_documents_chain(self.create_llm(), self.create_actual_system_prompt())
        rag_chain = create_retrieval_chain(self.create_history_retriever(), doc_chain)
        return rag_chain
    
    def create_session_history(self, session:str)->BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    def create_runnable_chat(self):
        runnable = RunnableWithMessageHistory(
            self.create_retrieval_chain(),
            self.create_session_history,
            input_messages_key="input",
            output_messages_key='answer',
            history_messages_key="chat_history"
        )
        return runnable
    
    def final_response(self, query, session_id):
        runnable = self.create_runnable_chat()
        response = runnable.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
        return response
    


st.title("Chat With Your PDF")
st.write("This is a chat interface that allows you to interact with a PDF document.")

# === Initialize session state variables only once ===
if "initialized" not in st.session_state:
    st.session_state.store = {}
    st.session_state.session_id = f"default_session-{''.join(random.choices(string.hexdigits, k=10))}"
    st.session_state.bot = None
    st.session_state.path = None
    st.session_state.initialized = True  # flag to prevent re-running

# === Session ID input ===
session_id = st.text_input("Session ID", value=st.session_state.session_id)
st.session_state.session_id = session_id  # allow user override

# === API key input ===
groq_api = st.text_input("Enter Groq API Key", type="password")

# === PDF Upload & Bot Initialization ===
if groq_api:
    pdf_file = st.file_uploader("Select a PDF file", type="pdf")
    if pdf_file and st.session_state.bot is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            st.session_state.path = temp_file.name

        # 🔧 Replace this with actual bot creation logic
        st.session_state.bot = RunnableChatBotWithHistory(groq_api, st.session_state.path)

# === Chat Interface ===
if st.session_state.bot:
    session_history = st.session_state.bot.create_session_history(st.session_state.session_id)
    user_input = st.text_input("You:")

    if user_input:
        response = st.session_state.bot.final_response(user_input, st.session_state.session_id)

        st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)



    


    
    



