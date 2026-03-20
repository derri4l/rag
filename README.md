# Sailor
Sailor is a RAG tool (Retrieval-Augmented Generation), it lets you load documents or notes, index them in memory, and ask questions. This is only intended for study purposes.

## How does it work ?
Pick a document/note from the knowledge folder. Then the pipline begins:
1. **Chunking** -  The note gets split into overlapping word chunks.
2. **Embeddings** - each chunk is converted into a vector using a local embedding model.
3. **FAISS Index** -  All vectors are stored in a FAISS flat index in memory.
4. **Retrieval** -  When you ask a question, it gets embedded the same way. FAISS finds the top 3 closest chunks by vector distance.
5. **Generation** -  The top chunks are passed to an LLM as context. The model answers using only what's in the retrieved chunks.

## Repo structure
```
sailor/
├── rag/
│   └── main.py          # main program
│   └── scrap.py         # web scraper
├── knowledge/           # your documents/notes (.txt)
├── requirements.txt
├── setup.py             # dependencies
└── README.md
```
 
 ## Installation  
 Requirements: Python 3.10+, Ollama installed and running.
 ```
to download the project:
> git clone https://github.com/derri4l/sailor.git

run this to get the dependencies installed:
> python3 setup.py

pull the default embedding model and LLM
> ollama pull nomic-embed-text
> ollama pull gemma3:1b

to use the tool:
> python3 rag/main.py
```

## Next Steps
- Auto chunk sizing [🗸]
- switching to local llm [🗸]
- Token-based chunking [ ]
- implement vector store/database [ ]
- TUI [ ]
