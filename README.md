**✅ Solution**

This project solves the problem by:
Extracting video transcripts automatically
Converting them into a searchable vector database
Using LLMs to answer questions based only on video content

**🧠 Key Highlights**
Implements end-to-end RAG pipeline
Uses semantic search instead of keyword search
Ensures context-aware answers
Fully local embeddings (cost-efficient)
Clean separation of frontend & backend

**⚙️ How It Works**
User enters a YouTube URL
Transcript is fetched using API
Text is split into chunks
Chunks are converted into embeddings
Stored in FAISS vector database
Relevant chunks retrieved based on query
LLM generates final answer

**📊 Core Components**
Data Ingestion: YouTube Transcript API
Processing: Text chunking (LangChain)
Embedding: Sentence Transformers
Storage: FAISS Vector DB
Retrieval: Similarity Search
Generation: Groq LLM

**🎯 Learning Outcomes**
Built a real-world RAG application
Understood vector databases & embeddings
Learned prompt engineering & retrieval optimization
Integrated LLM APIs in production-style apps
Designed scalable AI architecture
