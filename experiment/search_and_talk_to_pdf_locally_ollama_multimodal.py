import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the library root directory here
LIBRARY_ROOT = os.getenv('LIBRARY_ROOT')
PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE')

def search_pdfs(search_term):
    if not os.path.exists(LIBRARY_ROOT):
        print(f"Directory '{LIBRARY_ROOT}' does not exist.")
        return []
    result = []
    for root, dirs, files in os.walk(LIBRARY_ROOT):
        for filename in files:
            if filename.endswith('.pdf') and (search_term.lower() in filename.lower()):
                result.append(os.path.join(root, filename))
    return result

def select_pdf(pdf_list):
    if not pdf_list:
        print("No PDFs found matching the search term.")
        return None
    print("Found PDFs:")
    for i, pdf in enumerate(pdf_list, 1):
        print(f"{i}. {os.path.relpath(pdf, LIBRARY_ROOT)}")
    while True:
        try:
            choice = int(input("Enter the number of the PDF you want to use: "))
            if 1 <= choice <= len(pdf_list):
                return pdf_list[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def setup_conversation(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        # Use srizon/pixie for embeddings
        embeddings = OllamaEmbeddings(model="srizon/pixie")
        vectorstore = FAISS.from_documents(pages, embeddings)

        # Use srizon/pixie as the LLM
        llm = OllamaLLM(model="srizon/pixie")

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Example prompt template, set it in .env
        '''
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Chat History:
        {chat_history}
        H: {question}
        A:"""
        '''

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        return chain
    except Exception as e:
        print(f"Error setting up conversation: {e}")
        return None

def main():
    search_term = input("Enter a search term (title or author): ")
    pdf_list = search_pdfs(search_term)
    selected_pdf = select_pdf(pdf_list)

    if selected_pdf:
        conversation_chain = setup_conversation(selected_pdf)
        if conversation_chain:
            print(f"Conversation started for: {os.path.relpath(selected_pdf, LIBRARY_ROOT)}")
            print("Type 'exit' to end the conversation.")
            while True:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    break
                try:
                    response = conversation_chain({"question": user_input})
                    print("Assistant:", response['answer'])
                except Exception as e:
                    print(f"Error processing question: {e}")
        else:
            print("Failed to set up conversation. Please check your setup and try again.")

if __name__ == "__main__":
    main()
