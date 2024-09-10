import gradio as gr
from rag_chat_pinecone import Config, initialize_pinecone, get_pinecone_index, get_embeddings, create_vectorstore, get_llm, get_retriever, create_rag_chain, ask_question, PROMPT

config = Config()
pc = initialize_pinecone(config)
index = get_pinecone_index(pc, config)
embeddings = get_embeddings(config)
vectorstore = create_vectorstore(index, embeddings)
llm = get_llm(config)
retriever = get_retriever(vectorstore, None, llm)
rag_chain = create_rag_chain(llm, retriever, PROMPT)

conversation_history = []

def process_query(query, namespace):
    global conversation_history
    response = ask_question(query, namespace, rag_chain, vectorstore, llm)
    conversation_history.append((query, response))
    return format_conversation_history()

def format_conversation_history():
    formatted_history = ""
    for i, (q, a) in enumerate(conversation_history):
        formatted_history += f"Q{i+1}: {q}\n\nA{i+1}: {a}\n\n{'='*50}\n\n"
    return formatted_history

def get_namespaces():
    stats = index.describe_index_stats()
    return list(stats['namespaces'].keys()) + ['all']

with gr.Blocks() as demo:
    gr.Markdown("# RAG Pinecone Q&A System")

    namespace_dropdown = gr.Dropdown(choices=get_namespaces(), label="Select Namespace", value="all")
    conversation_history_output = gr.Textbox(label="Conversation History", lines=15, max_lines=15)

    with gr.Row():
        query_input = gr.Textbox(label="Enter your question", lines=2, placeholder="Type your question here...")
        submit_button = gr.Button("Submit")

    def update_conversation(query, namespace):
        return process_query(query, namespace)

    submit_button.click(
        update_conversation,
        inputs=[query_input, namespace_dropdown],
        outputs=[conversation_history_output]
    )

    # Clear the query input after submission
    submit_button.click(lambda: "", outputs=[query_input])

    # Update namespace when dropdown changes
    namespace_dropdown.change(
        lambda x: x,
        inputs=[namespace_dropdown],
        outputs=[namespace_dropdown]
    )

demo.launch()