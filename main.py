import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)
from typing import Any, Dict, List
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()


PDF_FOLDER = "C:\\Users\\itsra\\OneDrive\\Documents\\Workthon-deliverables\\code\\ingestion\\pdfs"
GPT_EMBEDDING_MODEL = str(os.environ.get("GPT_EMBEDDING_MODEL"))

embeddings = OpenAIEmbeddings(
    model=GPT_EMBEDDING_MODEL,
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=90,
)
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", retry_min_seconds=10)

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

def index_documents_async(documents: List[Document], batch_size: int = 20):
    """Process documents in batches."""
    log_header("VECTOR STORAGE PHASE")
    log_info(f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,)
    # Create batches
    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")
    # Process all batches in a sequence
    batch_num = 1
    for batch in batches:
        try:
            vectorstore.add_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
            batch_num += 1
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")

def main():
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "🗺️  Starting to crawl the pdf folder ",
        Colors.PURPLE,
    )
    all_chunks = []
    file_count = 0
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            print(f"Loading: {pdf_path}")

            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # Add filename + page number BEFORE chunking
            for page_num, page in enumerate(pages, start=1):
                page.metadata["source"] = file
                page.metadata["MY_page_number"] = page_num

            chunks = text_splitter.split_documents(pages)
            print(f" → {len(chunks)} chunks created")
            file_count+=1
            all_chunks.extend(chunks)

    log_success(
        f"Text Splitter: Created {len(all_chunks)} chunks from {file_count} files"
    )
    index_documents_async(all_chunks, batch_size=20)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_chunks)}")
    log_info(f"   • File count: {file_count}")


if __name__ == "__main__":
    main()