from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st
import tempfile
import random
import string
import os
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

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

chat_prompt_template = """
<Prompt>
  <Role>
    <Name>Personalized Learning Assistant</Name>
    <Description>You are a highly intelligent and friendly personalized learning assistant. Your role is to help students learn effectively by understanding and interpreting the uploaded document.</Description>
  </Role>

  <Goals>
    <Primary>Thoroughly read and analyze the uploaded document, and accurately answer any user questions based on its content.</Primary>
    <Secondary>Build trust with the user by providing clear, concise, and polite responses in a helpful and friendly tone. Ensure all answers are easy to understand and aligned with the user's learning needs.</Secondary>
  </Goals>

  <Instructions>
    <Instruction>Carefully read and interpret the entire uploaded document before responding.</Instruction>
    <Instruction>Respond only based on the document content. If a question is not addressed in the document, respond with: _"I'm sorry, I couldn't find information about this in the uploaded document."_</Instruction>
    <Instruction>When asked, provide a concise and accurate **summary** of the uploaded document.</Instruction>
    <Instruction>Create a **set of at least 10 multiple-choice questions (MCQs)**:
      • 5 easy and 5 hard questions  
      • Each question must have 4 answer options  
      • Highlight the correct answer in **bold**  
    </Instruction>
    <Instruction>Create **10 short answer questions** that cover key ideas and concepts from the entire document.</Instruction>
    <Instruction>Format your responses using proper Markdown:
      • Use **bold** for emphasis  
      • Use *italics* for definitions or notes  
      • Use bullet points or numbered lists for structured answers
    </Instruction>
    <Instruction>Maintain a friendly, professional, and encouraging tone throughout all responses.</Instruction>
  </Instructions>

  <Examples>
    <Example>**Q:** What is the main topic discussed in the document?  
**A:** The document primarily discusses *[insert main topic]*, explaining its key aspects and importance.</Example>

    <Example>**MCQ Example:**  
**Question:** What is the capital of France?  
- Berlin  
- Madrid  
- **Paris**  
- Rome</Example>

    <Example>**Short Answer Example:**  
**Question:** Define the term "photosynthesis."  
**Answer:** Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into food (glucose) and oxygen.</Example>
  </Examples>

  <Context>{context}</Context>
</Prompt>
"""

class StudentAssistant:
    def __init__(self, contextual_prompt, prompt_templete, file):
        self.file = file
        self.contextual_prompt = contextual_prompt
        self.prompt_templete = prompt_templete
        self.embeddings = self.return_embeddings()
        self.docs = self.loading_and_chunking()
        self.llm = self.return_llm()
        self.retriever = self.return_vector_store()
        self.rag_chain = self.return_chain()

    def loading_and_chunking(self):
        loader = PyPDFLoader(self.file)
        documents = loader.load()
        chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        return chunker.split_documents(documents=documents)
    
    def return_embeddings(self):
        return HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
    
    def return_llm(self):
        return ChatGroq(model="llama-3.1-8b-instant")
    
    def return_vector_store(self):
        vsdb = FAISS.from_documents(self.docs, self.embeddings)
        return vsdb.as_retriever()

    def create_chat_prompt_templetes(self):
        cp = ChatPromptTemplate.from_messages([
            ("system", self.contextual_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        sp = ChatPromptTemplate.from_messages([
            ("system", self.prompt_templete),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        return cp, sp

    def get_session_history(self, session:str)->BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    def return_chain(self):
        cp, sp = self.create_chat_prompt_templetes()
        history_retriever = create_history_aware_retriever(self.llm, self.retriever, cp)
        doc_chain = create_stuff_documents_chain(self.llm, sp)
        qa_chain = create_retrieval_chain(history_retriever, doc_chain)
        rag_chain = RunnableWithMessageHistory(
            qa_chain,
            self.get_session_history,
            input_messages_key="input",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )
        return rag_chain
    
    def return_response(self, query, session):
        resp = self.rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session}}
        )
        return resp['answer']

    



def initialize_session_state():
    if "initialized" not in st.session_state:
        st.session_state.store = {}
        st.session_state.bot = None
        st.session_state.path = None
        st.session_state.session_id = f"default_session-{''.join(random.choices(string.hexdigits, k=10))}"
        st.session_state.messages = []
        st.session_state.initialized = True

# Upload and save the PDF to a temporary path
def handle_file_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

# Render chat history
def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle user input and response generation
def handle_user_input():
    if prompt := st.chat_input("Ask me anything..."):
        # Show user input
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    session_history = st.session_state.session_id
                    response = st.session_state.bot.return_response(prompt, session_history)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main app
def main():
    initialize_session_state()

    st.title("🧠 Student Learning Assistant")
    st.markdown("Ask questions about your uploaded document or any general topic. I'm here to help you learn!")

    st.text_input("📌 Session ID", value=st.session_state.session_id, disabled=True)

    file_to_upload = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

    if file_to_upload:
        st.session_state.path = handle_file_upload(file_to_upload)

        # Initialize bot after file upload
        try:
            st.session_state.bot = StudentAssistant(
                contextual_prompt=contextualize_q_system_prompt,
                prompt_templete=chat_prompt_template,
                file=st.session_state.path
            )
            st.success("PDF loaded successfully. You can now ask questions.")

        except Exception as e:
            st.error(f"Failed to initialize assistant: {e}")
            return

    if st.session_state.bot:
        render_chat_history()
        handle_user_input()
    else:
        st.info("Please upload a PDF file to begin.")


main()