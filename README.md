# RAG Systems for Talking to Pinecone Vector Store & Search and Talk to Local Pdf Library

This repository contains a Retrieval-Augmented Generation (RAG) system that connect to your Pinecone vector storage for retrieval,
or searches your local pdf library to chat with your pdf files.
They both use Ollama to deploy dolphin-llama3:8b locally for unrestricted chat with your private knowledgebase.
To set up a pinecone index from pdf books u can look at my other project: "Pdf-extraction-to-vector-storage".


## Features

- Uses Pinecone for efficient vector storage and retrieval
- Employs Cohere for text embeddings
- Utilizes Ollama with dolphin-llama3:8b as the language model
- Implements a contextual compression retriever for more relevant document retrieval
- Supports namespaced queries in Pinecone
- Includes logging for better debugging and monitoring
- Search and talk directly to pdf's from 100% private library on local machine

## Requirements

- Python 3.7+
- Pinecone API key
- Cohere API key
- Ollama (locally installed)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Feed-dev/Library-chat.git
   
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_pinecone_index_name
   COHERE_API_KEY=your_cohere_api_key
   ```
4. Install the dolphin-llama3:8b:

   Install Ollama: https://ollama.com/

   dolphin-llama3:8b: https://ollama.com/library/dolphin-llama3:
   ```
   ollama pull dolphin-llama3
   ```

## Usage

Run the gradio ui script:

```
python rag_chat_pinecone_gradio_ui.py
```

Follow the prompts to select a namespace and enter your questions. The system will retrieve relevant context from Pinecone and generate answers using Ollama.

## Configuration

You can modify the following parameters in the `Config` class:
- `OLLAMA_MODEL`: The Ollama model to use (default is "dolphin-llama3:8b")
- Search parameters: Adjust the `search_kwargs` in the `get_retriever` function

## License

This project is licensed under the MIT License.
