from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00d4ff;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown(
    '<p class="title">🤖 RAG Document Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Ask Questions About Your Documents</p>',
    unsafe_allow_html=True
)

# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_rag():

    load_dotenv()

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="croma_db",
        embedding_function=embedding_model
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful AI assistant.

Use only the provided context.

If answer is not found, say:
"I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ])

    return retriever, llm, prompt


retriever, llm, prompt = load_rag()

# -----------------------
# Chat History
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# User Input
# -----------------------
query = st.chat_input("Ask something about your document...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching document..."):

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:

    st.title("⚙ Settings")

    st.markdown("---")

    st.write("Search Strategy")

    st.code("""
MMR Retriever
k = 4
fetch_k = 10
lambda_mult = 0.5
""")

    st.markdown("---")

    st.success("Vector DB Connected")