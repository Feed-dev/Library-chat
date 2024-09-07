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
    search_kwargs = {"k": 10}
    if namespace:
        search_kwargs["namespace"] = namespace
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    logger.info(f"Created base retriever with search_kwargs: {search_kwargs}")
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)


# Enhanced Prompt template
PROMPT_TEMPLATE = """Given the following context and question, provide a comprehensive and detailed answer. Explore the subject thoroughly, considering multiple aspects and perspectives if applicable. If the context doesn't contain enough information to fully answer the question, clearly state what is known based on the context and what additional information might be needed.

Context:
{context}

Question: {question}

Instructions:
1. Analyze the context thoroughly.
2. Answer the question comprehensively.
3. Provide relevant examples or explanations when possible.
4. If there are any uncertainties or gaps in the information, mention them.
5. Summarize key points at the end of your answer.

Detailed Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)


# Query expansion function
def expand_query(question: str, llm) -> str:
    expansion_prompt = f"""Expand the following question to improve search results. 
    Add relevant keywords and rephrase it to capture related concepts:

    Original question: {question}

    Expanded question:"""

    return llm(expansion_prompt)


# Answer formatting and summarization function
def format_and_summarize_answer(answer: str, llm) -> str:
    format_prompt = f"""Format and summarize the following answer. 
    Highlight key points, organize information clearly, and provide a brief summary:

    Original answer: {answer}

    Formatted and summarized answer:"""

    return llm(format_prompt)


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
def ask_question(question, namespaces, rag_chain, vectorstore, llm):
    try:
        preprocessed_question = simple_preprocess(question)
        expanded_question = expand_query(preprocessed_question, llm)
        #logger.info(f"Preprocessed query: {preprocessed_question}")
        #logger.info(f"Expanded query: {expanded_question}")

        all_docs = []
        if isinstance(namespaces, list):
            for namespace in namespaces:
                retriever = get_retriever(vectorstore, namespace, llm)
                #logger.info(f"Querying Pinecone in namespace '{namespace}' with: {expanded_question}")
                docs = retriever.get_relevant_documents(expanded_question)
                all_docs.extend(docs)
                #logger.info(f"Retrieved {len(docs)} documents from namespace '{namespace}'")
        else:
            retriever = get_retriever(vectorstore, namespaces, llm)
            #logger.info(f"Querying Pinecone in namespace '{namespaces}' with: {expanded_question}")
            all_docs = retriever.get_relevant_documents(expanded_question)
            #logger.info(f"Retrieved {len(all_docs)} documents from namespace '{namespaces}'")

        if len(all_docs) == 0:
            logger.warning("No documents retrieved. Consider adjusting search parameters or verifying content.")
            print("No relevant documents found. Try rephrasing your question or using different keywords.")
            return None

        '''
        logger.info("Retrieved documents:")
        for i, doc in enumerate(all_docs, 1):
            logger.info(f"Document {i}:")
            logger.info(f"Content: {doc.page_content[:200]}...")  # Log first 200 characters of content
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("---")
        '''

        # Use the rag_chain with the retrieved documents
        raw_response = rag_chain.invoke({"question": expanded_question, "context": all_docs})

        # Format and summarize the response
        formatted_response = format_and_summarize_answer(raw_response, llm)

        '''
        print("\nRetrieved Context:")
        for i, doc in enumerate(all_docs, 1):
            print(f"Document {i}:")
            print(doc.page_content)
            print("---")
        '''

        #print("\nFormatted Answer:", formatted_response)
        return formatted_response

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
        #logger.info(f"Index stats: {stats}")

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

            #print(f"\nProcessing question: {user_question}")
            print("\nExpanding query...")
            expanded_query = expand_query(user_question, llm)
            #print(f"Expanded query: {expanded_query}")

            print("\nRetrieving and processing information...")
            response = ask_question(expanded_query, namespaces, rag_chain, vectorstore, llm)

            if response:
                print("\nAnswer processed and formatted:")
                print(response)
                print("\nReady for next question.")
            else:
                print("\nAn error occurred while processing the question. Please try again.")

        print("Thank you for using the enhanced RAG system. Goodbye!")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Exception details:")


if __name__ == "__main__":
    main()
