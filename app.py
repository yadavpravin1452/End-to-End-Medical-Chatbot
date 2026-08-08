from flask import Flask, render_template, request, jsonify

from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embedding

from langchain_pinecone import PineconeVectorStore

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt


app = Flask(__name__)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")


embeddings = download_hugging_face_embedding()


docsearch = PineconeVectorStore.from_existing_index(
    index_name = PINECONE_INDEX_NAME,
    embedding = embeddings
)

TOP_K = int(os.getenv("TOP_K"))
retriever = docsearch.as_retriever(search_type = "similarity", 
                                   search_kwargs = {"k" : TOP_K})


LLM_TEMP = float(os.getenv("LLM_TEMPERATURE"))
MAX_TOKEN = int(os.getenv("MAX_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id = LLM_MODEL,
    huggingfacehub_api_token = HF_TOKEN,
    temperature = LLM_TEMP,
    max_new_tokens = MAX_TOKEN
)

chat_model = ChatHuggingFace(llm = llm)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                           ("human", "{input}"),])


question_answer_chain = create_stuff_documents_chain(chat_model, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_query = request.form["msg"]
    response = rag_chain.invoke({"input": user_query})
    return response["answer"]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)