from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_size"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


# Extract Data from PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    return loader.load()



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:

    '''
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and original page_content.
    '''

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source" : src}
            )
        )

    return minimal_docs


# Split the Data into Smaller Chunk
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk


# Download the Embedding from HuggingFace
def download_hugging_face_embedding():
    # Download and return the HuggingFace Embedding model
    embedding = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    return embedding

