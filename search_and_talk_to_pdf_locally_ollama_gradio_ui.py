import gradio as gr
import os
from search_and_talk_to_pdf_locally_ollama import search_pdfs, setup_conversation, LIBRARY_ROOT

conversation_chain = None
conversation_history = []
current_pdf = None


def gradio_interface(search_term, selected_pdf_path, user_input, action):
    global conversation_chain, conversation_history, current_pdf
    system_message = ""

    if action == "search":
        pdf_list = search_pdfs(search_term)
        if not pdf_list:
            system_message = "No PDFs found matching the search term."
            return system_message, gr.update(choices=[], value=None, visible=False), gr.update(
                visible=False), gr.update(visible=False), gr.update(value="")

        pdf_options = [os.path.relpath(pdf, LIBRARY_ROOT) for pdf in pdf_list]
        system_message = f"Found PDFs:\n" + "\n".join([f"{i + 1}. {pdf}" for i, pdf in enumerate(pdf_options)])
        return system_message, gr.update(choices=pdf_options, value=None, visible=True), gr.update(
            visible=True), gr.update(visible=True), gr.update(value="")

    elif action == "initiate" or action == "reinitiate":
        if selected_pdf_path is None:
            system_message = "Please select a PDF first."
            return system_message, gr.update(value=None), gr.update(visible=True), gr.update(visible=True), gr.update(
                value="\n".join(conversation_history))

        selected_pdf = os.path.join(LIBRARY_ROOT, selected_pdf_path)

        if action == "reinitiate" and current_pdf:
            system_message = f"Continuing conversation. Previous PDF: {os.path.relpath(current_pdf, LIBRARY_ROOT)} is no longer available."

        conversation_chain = setup_conversation(selected_pdf)
        current_pdf = selected_pdf

        if not conversation_chain:
            system_message = "Failed to set up conversation. Please check your setup and try again."
            return system_message, gr.update(value=selected_pdf_path), gr.update(visible=True), gr.update(
                visible=True), gr.update(value="\n".join(conversation_history))

        system_message += f"\nConversation initiated for: {selected_pdf_path}\nYou can now start asking questions."
        return system_message, gr.update(value=selected_pdf_path), gr.update(visible=False), gr.update(
            visible=True), gr.update(value="\n".join(conversation_history))

    elif action == "chat":
        if not conversation_chain:
            system_message = "Please initiate the conversation first."
            return system_message, gr.update(value=selected_pdf_path), gr.update(visible=True), gr.update(
                visible=True), gr.update(value="\n".join(conversation_history))

        if user_input.lower() == 'exit':
            system_message = "Conversation ended."
            return system_message, gr.update(value=selected_pdf_path), gr.update(visible=True), gr.update(
                visible=True), gr.update(value="\n".join(conversation_history))

        try:
            response = conversation_chain({"question": user_input})
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response['answer']}")
            return gr.update(value=""), gr.update(value=selected_pdf_path), gr.update(visible=False), gr.update(
                visible=True), gr.update(value="\n".join(conversation_history))
        except Exception as e:
            system_message = f"Error processing question: {e}"
            return system_message, gr.update(value=selected_pdf_path), gr.update(visible=False), gr.update(
                visible=True), gr.update(value="\n".join(conversation_history))


with gr.Blocks() as demo:
    gr.Markdown("# PDF Search and Chat")

    with gr.Row():
        search_input = gr.Textbox(label="Enter a search term (title or author)")
        search_button = gr.Button("Search")

    pdf_dropdown = gr.Dropdown(choices=[], label="Select a PDF", visible=False)
    initiate_button = gr.Button("Initiate Conversation", visible=False)
    reinitiate_button = gr.Button("Reinitiate Conversation with Selected PDF", visible=False)

    system_output = gr.Textbox(label="System Messages", lines=3)
    conversation_output = gr.Textbox(label="Conversation History", lines=15, max_lines=30)

    with gr.Row():
        user_input = gr.Textbox(label="You", placeholder="Type your question here...", scale=4)
        submit_button = gr.Button("Submit", scale=1)


    def search_and_update(search_term):
        return gradio_interface(search_term, None, "", "search")


    search_button.click(search_and_update, inputs=[search_input],
                        outputs=[system_output, pdf_dropdown, initiate_button, reinitiate_button, conversation_output])


    def initiate_conversation(search_term, selected_pdf_path):
        return gradio_interface(search_term, selected_pdf_path, "", "initiate")


    initiate_button.click(initiate_conversation, inputs=[search_input, pdf_dropdown],
                          outputs=[system_output, pdf_dropdown, initiate_button, reinitiate_button,
                                   conversation_output])


    def reinitiate_conversation(search_term, selected_pdf_path):
        return gradio_interface(search_term, selected_pdf_path, "", "reinitiate")


    reinitiate_button.click(reinitiate_conversation, inputs=[search_input, pdf_dropdown],
                            outputs=[system_output, pdf_dropdown, initiate_button, reinitiate_button,
                                     conversation_output])


    def process_input(search_term, selected_pdf_path, user_input):
        return gradio_interface(search_term, selected_pdf_path, user_input, "chat")


    submit_button.click(process_input, inputs=[search_input, pdf_dropdown, user_input],
                        outputs=[user_input, pdf_dropdown, initiate_button, reinitiate_button, conversation_output])

demo.launch()
