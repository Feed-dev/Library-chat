# main.py

from rag_system import initialize_vector_store, initialize_llm, create_rag_chain, get_answer


def main():
    print("Initializing RAG system...")
    vector_store = initialize_vector_store()
    llm = initialize_llm()
    qa_chain = create_rag_chain(vector_store, llm)
    print("RAG system initialized. Ready for questions!")

    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        answer, sources = get_answer(qa_chain, question)
        print(f"\nAnswer: {answer}")
        print("\nSources:")
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source.page_content[:100]}...")

        # Display chat history
        display_chat_history(qa_chain)

    print("Thank you for using the RAG system!")

def display_chat_history(qa_chain):
    history = qa_chain.memory.chat_memory.messages
    print("\nChat History:")
    for message in history:
        if message.type == "human":
            print(f"Human: {message.content}")
        elif message.type == "ai":
            print(f"Assistant: {message.content}")
    print()

if __name__ == "__main__":
    main()
