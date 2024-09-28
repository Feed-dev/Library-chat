import os
from langchain.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# **Configuration**
LLAMA_MODEL_PATH = "path/to/llama-3.2-model.bin"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "faiss_index"

# **Initialize Llama**
llm = LlamaCpp(model_path=LLAMA_MODEL_PATH,
               n_ctx=2048,
               temperature=0.7,
               max_tokens=512)

# **Initialize Embeddings**
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# **Load or Create Vector Store**
if os.path.exists(VECTOR_STORE_PATH):
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
else:
    # Example: Indexing documents from a directory
    from langchain.document_loaders import DirectoryLoader
    loader = DirectoryLoader("path/to/documents", glob="**/*.txt")
    documents = loader.load()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

# **Initialize Retrieval QA**
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

def answer_query(query: str) -> str:
    """Generates an answer to the given query using the RAG agent."""
    return qa.run(query)

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    response = answer_query(user_query)
    print("Answer:", response)
