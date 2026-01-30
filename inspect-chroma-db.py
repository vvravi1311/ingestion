# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os

PDF_FOLDER = "C:\\Users\\itsra\\OneDrive\\Documents\\Workthon-deliverables\\code\\ingestion\\pdfs"
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
GPT_EMBEDDING_MODEL = os.environ.get("GPT_EMBEDDING_MODEL")
SETENCE_TRANSFOMER_EMBEDDING_MODEL = os.environ.get("SETENCE_TRANSFORMER_EMBEDDING_MODEL")
CHROMA_DIRECTORY=os.environ.get("CHROMA_DIRECTORY")
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    collection_name="medicare-docs",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIRECTORY
)

print("Number of vectors:", db._collection.count())
