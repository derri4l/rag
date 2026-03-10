# rag
This project uses Python to create a RAG tool that lets you parse notes and query notes. This was only intended for study purposes.

## How does it work
Pick a note from the knowledge folder. Then the pipline begins:
1. Chunking - The note gets split into overlapping word chunks.
2. Embeddings - Each chunk is sent to OpenAI's text-embedding-3-small model and converted into a vector.
3. FAISS Index - All vectors are stored in a FAISS flat index in memory.
4. Retrieval - When you ask a question, it gets embedded the same way. FAISS finds the top 3 closest chunks by vector distance.
5. Generation - The top chunks are passed to gpt4o mini as context. The model answers using only what's in the retrieved chunks.


## How to use
1. install dependencies  
```pip install openai faiss-cpu numpy python-dotenv requests```

2. Add your OpenAI key to a .env file:
```OPENAI_API_KEY=key```

3. Run with
```python3 main.py```
