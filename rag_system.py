# rag_system.py

from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import config

def initialize_vector_store():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=config.COHERE_API_KEY)
    return LangchainPinecone(pinecone_index, embeddings, "text")

def initialize_llm():
    return Ollama(model="dolphin-llama3:8b", base_url=config.OLLAMA_BASE_URL)

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