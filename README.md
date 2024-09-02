# RAG System with Pinecone and Ollama

This repository contains a Retrieval-Augmented Generation (RAG) system that connect to your Pinecone vector storage for retrieval, 
and Ollama to deploy dolphin-llama3:8b locally for unrestricted chat with your private vectorized library. 
The system is designed to answer questions based on a given context, which is retrieved from the Pinecone index.
To set up a pinecone index from pdf books u can look at my other project: "Pdf-extraction-to-vector-storage".

## Features

- Uses Pinecone for efficient vector storage and retrieval
- Employs Cohere for text embeddings
- Utilizes Ollama with dolphin-llama3:8b as the language model
- Implements a contextual compression retriever for more relevant document retrieval
- Supports namespaced queries in Pinecone
- Includes logging for better debugging and monitoring

## Requirements

- Python 3.7+
- Pinecone API key
- Cohere API key
- Ollama (locally installed)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Feed-dev/Library-chat.git
   cd rag-system
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

## Usage

Run the main script:

```
python rag_system_1.1.py
```

Follow the prompts to select a namespace and enter your questions. The system will retrieve relevant context from Pinecone and generate answers using Ollama.

## Configuration

You can modify the following parameters in the `Config` class:

- `OLLAMA_MODEL`: The Ollama model to use (default is "dolphin-llama3:8b")
- Search parameters: Adjust the `search_kwargs` in the `get_retriever` function

## License

This project is licensed under the MIT License.
