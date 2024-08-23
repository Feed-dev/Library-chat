# rag_system.py

from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

def initialize_vector_store():
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    embeddings = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=COHERE_API_KEY,
        client=None
    )
    return LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)

def initialize_llm():
    return Ollama(model="dolphin-llama3:8b", base_url=OLLAMA_BASE_URL or "http://localhost:11434")

def create_rag_chain(vector_store, llm):
    prompt_template = """You are a highly capable research assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

def get_answer(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"], result["source_documents"]