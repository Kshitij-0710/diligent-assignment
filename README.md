# 🤖 Jarvis - AI Assistant with Long-term Memory

A conversational AI assistant built with Streamlit, Ollama, and Pinecone that features persistent memory capabilities through vector storage.

## Features

- **Conversational Interface**: Clean, user-friendly chat interface powered by Streamlit
- **Local LLM**: Uses Ollama's Llama 3.2 (1B) model for fast, privacy-focused responses
- **Long-term Memory**: Automatically stores conversation context in Pinecone vector database
- **RAG (Retrieval-Augmented Generation)**: Retrieves relevant context from past conversations to provide informed responses
- **Chat History**: Maintains conversation continuity within the session

## Architecture

```
User Input → Pinecone (Auto-save) → Retriever → LLM → Response
                ↓
          Vector Embeddings
```

The application:
1. Automatically saves user inputs to Pinecone for long-term memory
2. Retrieves relevant context from the vector store
3. Uses LangChain to combine context with chat history
4. Generates contextually-aware responses using Ollama

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Pinecone account and API key
- Llama 3.2:1b model pulled in Ollama

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd diligent-assignment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama**
   ```bash
   # Install Ollama from https://ollama.ai/
   # Pull the required model
   ollama pull llama3.2:1b
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

5. **Create Pinecone index**
   ```bash
   python create_index.py
   ```

## Usage

1. **Start the application**
   ```bash
   streamlit run main.py
   ```

2. **Access the interface**
   
   Open your browser and navigate to `http://localhost:8501`

3. **Start chatting**
   
   Type your messages in the chat input. Jarvis will respond based on:
   - Current conversation context
   - Previously stored conversations in Pinecone
   - Chat history from the current session

## Project Structure

```
diligent-assignment/
├── main.py              # Main Streamlit application
├── create_index.py      # Pinecone index initialization script
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not tracked)
└── README.md           # This file
```

## Configuration

### Pinecone Settings
- **Index Name**: `jarvis-demo`
- **Dimension**: 2048 (matches Ollama embedding dimension)
- **Metric**: Cosine similarity
- **Cloud**: AWS (us-east-1)

### Model Settings
- **LLM**: Ollama Llama 3.2:1b
- **Embeddings**: Ollama Llama 3.2:1b

## How It Works

1. **User Input Processing**:
   - User message is displayed immediately
   - Message is automatically embedded and stored in Pinecone

2. **Context Retrieval**:
   - Relevant past conversations are retrieved from Pinecone
   - Chat history from current session is maintained

3. **Response Generation**:
   - LangChain combines retrieved context with current query
   - Ollama LLM generates contextually-aware response
   - Response is displayed and added to chat history

## Notes

- This is a **class assignment** project using fictional data for testing
- The application gracefully handles errors to maintain smooth chat experience
- All conversations are stored in the vector database for future context

## Troubleshooting

**Error: PINECONE_API_KEY not found**
- Ensure `.env` file exists with valid Pinecone API key

**Error: Connection refused (Ollama)**
- Make sure Ollama is running: `ollama serve`
- Verify the model is installed: `ollama list`

**Error: Index not found**
- Run `python create_index.py` to create the Pinecone index

## Dependencies

- `streamlit` - Web interface
- `langchain` - LLM orchestration framework
- `langchain-community` - Community integrations
- `langchain-pinecone` - Pinecone vector store integration
- `pinecone-client` - Pinecone SDK
- `ollama` - Local LLM runtime
- `python-dotenv` - Environment variable management

## License

This project is for educational purposes.
