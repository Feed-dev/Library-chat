# rag_system.py

from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
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
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify the output key to use for memory
    )

    prompt_template = """You are a highly capable research assistant. Use the following pieces of context and the chat history to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Chat History: {chat_history}

    Human: {question}

    Assistant: """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        output_key="answer"  # Specify the output key here as well
    )

def get_answer(qa_chain, question):
    try:
        result = qa_chain({"question": question})
        return result["answer"], result.get("source_documents", [])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question.", []
