# 📚 LangChain-RAG-PDF-Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot that enables intelligent Q&A over your PDF documents using state-of-the-art NLP models.

## 🌟 Features

- **📄 PDF Processing**: Automatically extracts and processes text from multiple PDF documents
- **🧠 Semantic Search**: Uses advanced embeddings for accurate context retrieval
- **💬 Natural Conversations**: Interactive chat interface powered by BLOOM LLM
- **⚡ Efficient**: Optimized chunking and retrieval for fast responses
- **🎨 User-Friendly**: Clean Gradio interface with real-time interactions

## 🏗️ Architecture

The system follows a 5-stage pipeline:

**Document Loading → Text Chunking → Embedding & Storage → LLM Processing → User Interface**

### Pipeline Stages

1. **Document Loading** 
   - PyPDFLoader extracts text from PDF files
   - Processes multiple documents from a directory

2. **Text Chunking**
   - Splits documents into 800-character chunks
   - 150-character overlap for context continuity
   - Maintains document metadata

3. **Embedding & Vectorization**
   - Uses sentence-transformers all-MiniLM-L6-v2
   - Stores vectors in FAISS for efficient similarity search
   - Retrieves top-3 most relevant chunks

4. **LLM Processing**
   - BigScience BLOOM-560M model
   - Context-aware response generation
   - Temperature: 0.3 for focused answers

5. **User Interface**
   - Gradio-powered chat interface
   - Real-time question answering
   - Chat history management

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. Clone the repository

2. Install dependencies

3. Create a data directory name it "cours" and add your PDF files

## 💻 Usage

### Basic Usage

Run the notebook or Python script. The Gradio interface will launch automatically. You can then:
1. Ask questions about your documents
2. Get contextual answers based on the content
3. View chat history

### Configuration

Key parameters you can adjust:

- **Chunking parameters**: chunk_size (800), overlap (150)
- **Retrieval parameters**: k (3 chunks to retrieve)
- **LLM parameters**: max_new_tokens (256), temperature (0.3), top_p (0.95)

## 📊 Performance Metrics

| Metric               | Value  |
|----------------------|--------|
| Pages Processed      | 206    |
| Text Chunks          | 176    |
| Embedding Dimensions | 384    |
| Max Output Tokens    | 256    |
| Retrieval Time       | ~100ms |
| Generation Time      | ~2-3s  |

## ⚠️ Known Limitations

**Important Note**: This is an experimental RAG system with some limitations:

- **Response Quality**: The system uses BLOOM-560M, a relatively small language model. While it works well for many queries, it may occasionally produce:
  - Incomplete or truncated answers
  - Grammatically incorrect responses
  - Hallucinated information not present in the documents
  - Context misunderstandings

- **Model Constraints**: The lightweight BLOOM-560M model was chosen for accessibility and speed, but it has limitations compared to larger models like GPT-3.5 or GPT-4.

- **Recommendations for Better Performance**:
  - Use larger models (BLOOM-1b7 or BLOOM-3b) if you have sufficient GPU memory
  - Consider using more advanced models like Mistral-7B or Llama-2-7B
  - Adjust temperature and top_p parameters for more consistent outputs
  - Provide more specific and focused questions

**This project is intended for educational and experimental purposes.** For production use cases requiring high accuracy, consider upgrading to more powerful language models or commercial APIs.

## 🛠️ Technology Stack

### Core Libraries

**Document Processing**
- PyPDF - PDF text extraction
- LangChain - Document loading & processing

**Embeddings**
- sentence-transformers - Text embeddings
- all-MiniLM-L6-v2 - Compact & efficient model

**Vector Database**
- FAISS - Fast similarity search
- CPU-optimized implementation

**Language Model**
- transformers - HuggingFace integration
- bigscience/bloom-560m - Multilingual LLM

**User Interface**
- Gradio - Interactive web interface
- Real-time chat functionality

## 📖 How It Works

### 1. Document Ingestion

The system loads all PDF files from the specified directory and extracts their text content.

### 2. Intelligent Chunking

Documents are split into manageable chunks with overlap to maintain context across boundaries.

### 3. Vector Search & Retrieval

Text chunks are converted to embeddings and stored in FAISS for efficient similarity search.

### 4. RAG Response Generation

When a query is received:
- Relevant chunks are retrieved from the vector database
- Context is built from the top matches
- The LLM generates a response based on the context
- The answer is formatted and returned to the user

## 🐛 Troubleshooting

### Common Issues

**Poor Answer Quality / Erroneous Outputs**
- **Most Common Issue**: The BLOOM-560M model may produce inconsistent or incorrect responses
- Try increasing the chunk_size to provide more context
- Adjust temperature (lower values like 0.1-0.2 = more focused and deterministic)
- Consider upgrading to a larger, more capable model
- Rephrase your question to be more specific
- Note: Small models like BLOOM-560M are prone to errors and should be used with caution

**Model-Specific Errors**
- **Hallucinations**: The model may generate plausible-sounding but incorrect information
- **Incomplete Responses**: Short or cut-off answers are common with small models
- **Inconsistent Quality**: Response quality can vary significantly between queries
- **Solution**: Upgrade to models like Mistral-7B, Llama-2-7B, or use commercial APIs for production use

## 📝 Example Queries

- What are the main topics covered in the document?
- Explain the concept of [specific term]
- What is the relationship between [X] and [Y]?
- Summarize the section about [topic]

## 🙏 Technics/technologies

- HuggingFace for models and transformers
- LangChain for RAG framework
- FAISS for vector search
- Gradio for the UI framework

---

⭐Thank you.
