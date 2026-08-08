import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_file, filter_to_minimal_docs,
    text_split, download_hugging_face_embedding)


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION"))
PINECONE_METRICS = os.getenv("PINECONE_METRICS")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")



extracted_data = load_pdf_file(data = "data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunk = text_split(filter_data)

embeddings = download_hugging_face_embedding()


pc = Pinecone(api_key = PINECONE_API_KEY)


index_name = PINECONE_INDEX_NAME
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION"))
PINECONE_METRICS = os.getenv("PINECONE_METRICS")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
PINECONE_REGION = os.getenv("PINECONE_REGION")


if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = PINECONE_DIMENSION,
        metric = PINECONE_METRICS,
        spec = ServerlessSpec(cloud = PINECONE_CLOUD, region = PINECONE_REGION)
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents = text_chunk,
    index_name = PINECONE_INDEX_NAME,
    embedding = embeddings,
    pinecone_api_key = PINECONE_API_KEY
)