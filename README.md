# rag
This project uses Python to create a RAG tool that parses notes. This was only intended for study purposes.

## How does it work
- First the program indexes your notes (.txt or .md).
- The notes gets chunked and turned into embeddings. During this process overlap is implemented so we don't loose context.
- Stores the vectors using FAISS.
- When we query now, the program retrieves the top 3 relevant chunks.
- Finally those top chunks get passed to an LLM to generate the final answer.

## Diagram


## How to use

