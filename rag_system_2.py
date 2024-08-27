import os
import logging
import string
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.schema.runnable import RunnablePassthrough

# Configuration
class Config:
    def __init__(self):
        load_dotenv()
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        self.OLLAMA_MODEL = "dolphin-llama3:8b"

# Logging setup
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

# Pinecone utilities
def initialize_pinecone(config):
    return Pinecone(api_key=config.PINECONE_API_KEY)

def get_pinecone_index(pc, config):
    return pc.Index(config.PINECONE_INDEX_NAME)

# Custom Cohere Embeddings
class CustomCohereEmbeddings(CohereEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ensure each text is a string
        texts = [str(text) for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        # Ensure the query is a string
        return super().embed_query(str(text))

# Embedding setup
def get_embeddings(config) -> Embeddings:
    return CustomCohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=config.COHERE_API_KEY
    )

# Vector store setup
def create_vectorstore(index, embeddings):
    return PineconeVectorStore(index, embeddings, text_key="text")

# LLM setup
def get_llm(config):
    return Ollama(
        model=config.OLLAMA_MODEL,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

# Retriever setup
def get_retriever(vectorstore, namespace, llm):
    search_kwargs = {"k": 10}  # Removed score_threshold
    if namespace:
        search_kwargs["namespace"] = namespace
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    logger.info(f"Created base retriever with search_kwargs: {search_kwargs}")
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

# Simplified Prompt template
PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question based on the provided context."""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Create RAG chain
def create_rag_chain(llm, retriever, prompt):
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm
    )
    return rag_chain

# Text preprocessing
def simple_preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

# Question answering function
def ask_question(question, namespace, rag_chain, vectorstore, llm):
    try:
        preprocessed_question = simple_preprocess(question)
        logger.info(f"Preprocessed query: {preprocessed_question}")

        # Create a new retriever for each question
        retriever = get_retriever(vectorstore, namespace, llm)

        logger.info(
            f"Querying Pinecone {('in namespace ' + namespace) if namespace else ''} with: {preprocessed_question}")

        try:
            # Use the invoke method instead of get_relevant_documents
            retriever_output = retriever.invoke(preprocessed_question)

            # Check if retriever_output is a list or a dict
            if isinstance(retriever_output, list):
                docs = retriever_output
            elif isinstance(retriever_output, dict) and 'documents' in retriever_output:
                docs = retriever_output['documents']
            else:
                logger.error(f"Unexpected retriever output format: {type(retriever_output)}")
                return None

            logger.info(f"Retrieved {len(docs)} documents from Pinecone")

            if len(docs) == 0:
                logger.warning("No documents retrieved. Consider adjusting search parameters or verifying content.")
                print("No relevant documents found. Try rephrasing your question or using different keywords.")
                return None

            logger.info("Retrieved documents:")
            for i, doc in enumerate(docs, 1):
                logger.info(f"Document {i}:")
                logger.info(f"Content: {doc.page_content[:200]}...")  # Log first 200 characters of content
                logger.info(f"Metadata: {doc.metadata}")
                logger.info("---")

            # Now use the rag_chain with the retrieved documents
            response = rag_chain.invoke({"question": preprocessed_question})

            print("\nRetrieved Context:")
            for i, doc in enumerate(docs, 1):
                print(f"Document {i}:")
                print(doc.page_content)
                print("---")

            print("\nAnswer:", response)

            return response
        except Exception as e:
            logger.error(f"Error during retrieval or processing: {e}")
            logger.exception("Exception details:")
            print(f"An error occurred while processing the question: {e}")
            return None
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        logger.exception("Exception details:")
        print(f"An unexpected error occurred: {e}")
        return None

# Main execution
def main():
    config = Config()
    
    try:
        pc = initialize_pinecone(config)
        index = get_pinecone_index(pc, config)
        logger.info(f"Successfully connected to Pinecone index: {config.PINECONE_INDEX_NAME}")
        
        # Get and log index stats
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        if stats['total_vector_count'] == 0:
            logger.warning("The index is empty. Make sure you've populated it with vectors.")
            return

        embeddings = get_embeddings(config)
        vectorstore = create_vectorstore(index, embeddings)
        llm = get_llm(config)
        retriever = get_retriever(vectorstore, None, llm)
        rag_chain = create_rag_chain(llm, retriever, PROMPT)
        
        print("\nAvailable namespaces:")
        for namespace, ns_stats in stats['namespaces'].items():
            print(f"- {namespace} ({ns_stats['vector_count']} vectors)")
        print("- [none] (use total vector store)")
        
        print("\nEnter the namespace to search (or press Enter to use 'philosophy'):")
        namespace = input().strip() or "philosophy"
        print(f"Selected namespace: {namespace}")
        
        while True:
            print("\nEnter your question (or 'quit' to exit):")
            user_question = input()
            if user_question.lower() == 'quit':
                break
            print(f"Processing question: {user_question}")
            response = ask_question(user_question, namespace, rag_chain, vectorstore, llm)
            if response:
                print("\nAnswer processed. Ready for next question.")
            else:
                print("\nAn error occurred while processing the question. Please try again.")
        
        print("Thank you for using the RAG system. Goodbye!")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Exception details:")

if __name__ == "__main__":
    main()
