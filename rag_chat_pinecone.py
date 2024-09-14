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

# Global variable for conversation history
conversation_history = []

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
        texts = [str(text) for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
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
    search_kwargs = {"k": 10}
    if namespace:
        search_kwargs["namespace"] = namespace
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    logger.info(f"Created base retriever with search_kwargs: {search_kwargs}")
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

# Updated Prompt template
PROMPT_TEMPLATE = """
You are an expert knowledge base researcher and organizer.

Instructions:
1. Thoroughly analyze the provided context as an expert in knowledge base management and information architecture.
2. Gather all question-related relevant information from the context that can contribute to a comprehensive overview of the subject.
3. List the titles and authors of any referenced sources or documents for citation purposes.
4. Examine the conversation history ONLY if the question explicitly refers to it.
5. Organize the information in a clear, logical structure using appropriate headers and subheaders.
6. Provide concise yet detailed explanations, avoiding unnecessary jargon

Conversation history:
{conversation_history}

Context:
{context}

Question:
{question}

Response:
[Your structured and detailed response here, following the instructions above]

"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question", "conversation_history"]
)

# Create RAG chain
def create_rag_chain(llm, retriever, prompt):
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "conversation_history": lambda x: "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history[-3:]])
        }
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
def ask_question(question, namespaces, rag_chain, vectorstore, llm):
    global conversation_history
    try:
        preprocessed_question = simple_preprocess(question)
        logger.info(f"Preprocessed query: {preprocessed_question}")

        all_docs = []
        if isinstance(namespaces, list):
            for namespace in namespaces:
                retriever = get_retriever(vectorstore, namespace, llm)
                logger.info(f"Querying Pinecone in namespace '{namespace}' with: {preprocessed_question}")
                docs = retriever.get_relevant_documents(preprocessed_question)
                all_docs.extend(docs)
                logger.info(f"Retrieved {len(docs)} documents from namespace '{namespace}'")
        else:
            retriever = get_retriever(vectorstore, namespaces, llm)
            logger.info(f"Querying Pinecone in namespace '{namespaces}' with: {preprocessed_question}")
            all_docs = retriever.get_relevant_documents(preprocessed_question)
            logger.info(f"Retrieved {len(all_docs)} documents from namespace '{namespaces}'")

        if len(all_docs) == 0:
            logger.warning("No documents retrieved. Consider adjusting search parameters or verifying content.")
            print("No relevant documents found. Try rephrasing your question or using different keywords.")
            return None

        logger.info("Retrieved documents:")
        for i, doc in enumerate(all_docs, 1):
            logger.info(f"Document {i}:")
            logger.info(f"Content: {doc.page_content[:200]}...") # Log first 200 characters of content
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("---")

        # Prepare conversation history
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history[-3:]])

        # Use the rag_chain with the retrieved documents and conversation history
        response = rag_chain.invoke({
            "question": preprocessed_question,
            "context": all_docs,
            "conversation_history": history_text
        })

        # Add the current Q&A to the conversation history
        conversation_history.append((question, response))

        print("\nRetrieved Context:")
        for i, doc in enumerate(all_docs, 1):
            print(f"Document {i}:")
            print(doc.page_content)
            print("---")

        print("\nAnswer:", response)
        return response

    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        logger.exception("Exception details:")
        print(f"An unexpected error occurred: {e}")
        return None

# Main execution
def main():
    global conversation_history
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
        print("- [all] (use all namespaces)")

        print("\nEnter the namespace(s) to search (comma-separated, or 'all' for all namespaces):")
        namespace_input = input().strip().lower()
        if namespace_input == 'all':
            namespaces = list(stats['namespaces'].keys())
        elif ',' in namespace_input:
            namespaces = [ns.strip() for ns in namespace_input.split(',')]
        else:
            namespaces = namespace_input

        print(f"Selected namespace(s): {namespaces}")

        while True:
            print("\nEnter your question (or 'quit' to exit):")
            user_question = input()
            if user_question.lower() == 'quit':
                break

            print(f"Processing question: {user_question}")
            response = ask_question(user_question, namespaces, rag_chain, vectorstore, llm)

            if response:
                print("\nAnswer processed. Ready for next question.")
            else:
                print("\nAn error occurred while processing the question. Please try again.")

        print("Thank you for using the RAG system. Goodbye!")
        conversation_history = []  # Clear conversation history at the end of the session

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Exception details:")

if __name__ == "__main__":
    main()
