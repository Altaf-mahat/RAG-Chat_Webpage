from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()

loader = PyPDFLoader(r"Inception_Movie_Information.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectore_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="croma_db"
)

result = vectore_store.similarity_search(
    query="What is movie name?",
    k=1
)

for i in result:
    print(i.page_content)
    print(i.metadata)




