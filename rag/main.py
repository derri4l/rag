import os
import sys
import textwrap

import faiss
import numpy as np
import ollama

knowledge_dir = os.path.join(os.path.dirname(__file__), "../knowledge/")


# List all txt notes in knowledge
def list_notes(folder=knowledge_dir):
    files = [f for f in os.listdir(folder) if f.endswith((".txt"))]
    for i, f in enumerate(files):
        print(f"{i + 1}. {f}")
    return files


# pick note
def select_note(files):
    while True:
        try:
            choice = int(input("\nSelect a note (use numbers): ")) - 1
            if 0 <= choice < len(files):
                return files[choice]
            print(f"  Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("  Invalid input, enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            sys.exit(0)


# read note
def load_note(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# detect and set overlap/chunk size
def detect_chunk(text: str) -> tuple[int, int]:
    word_count = len(text.split())
    if word_count < 500:
        return 150, 40
    elif word_count < 5000:
        return 300, 80
    else:
        return 500, 100


# split text into chunks w overlap
def chunk_text(
    text: str, chunk_size: int = 300, overlap: int = 80
) -> list[str]:  # change based on note length
    if overlap >= chunk_size:
        raise ValueError('"Overlap must be smaller than the chunk"')
    words = text.split()
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    vectors = []
    for chunk in chunks:
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        vectors.append(response["embedding"])
    return vectors


# store returned vectors in FAISS index
def store_in_faiss(vectors: list[list[float]]):
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))
    return index


def build_context(chunks: list[str]) -> str:
    return "\n\n".join(chunks)


# LLM prompt
def build_prompt(context: str, question: str) -> str:
    return f"""
    Only answer questions using the provided notes.

    Context:
    {context}

    Question:
    {question}
"""


def generate_answer(prompt: str) -> str:
    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 500},
    )
    return response["message"]["content"]


def print_answer(answer: str):
    formatted = format_answer(answer)
    for line in formatted.split("\n"):
        if line.startswith("  •") or line.startswith("  $") or line.startswith('  "'):
            print(line)
        else:
            print(textwrap.fill(line, width=80) if line.strip() else "")


def format_answer(answer: str) -> str:
    lines = answer.split("\n")
    formatted = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```") or stripped.endswith("```"):
            clean = stripped.replace("`", "").strip()
            formatted.append(f"$ {clean}")

        elif stripped.startswith("`") and stripped.endswith("`"):
            formatted.append(f"$ {stripped.strip('`')}")

        # blockquotes
        elif stripped.startswith(">"):
            content = stripped.lstrip("> ").strip()
            formatted.append(f'  "{content}"')

        # unordered lists
        elif stripped.startswith(("-", "*", "•")):
            content = stripped.lstrip("-*• ").strip()
            formatted.append(f"  • {content}")

        # numbered lists
        elif stripped and stripped[0].isdigit() and "." in stripped[:3]:
            formatted.append(f"  {stripped}")

        else:
            formatted.append(line)

    return "\n".join(formatted)


# embed query, search FAISS, return top 3 matching chunks
def search(query: str, index, chunks: list[str], top_k: int = 3):
    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    # convert query to vector, find the distance and position
    query_vector = np.array([response["embedding"]], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)

    # use positions to find chunk
    results = []
    for rank, (i, dist) in enumerate(zip(indices[0], distances[0])):
        # print(f"result {rank + 1} — distance: {dist:.4f}")
        results.append(chunks[i])
    return results


def main():
    print("\n-------------------- Notes -------------------")
    files = list_notes()
    selected = select_note(files)
    text = load_note(os.path.join(knowledge_dir, selected))

    # detect chunks
    chunk_size, overlap = detect_chunk(text)
    print(f"Word count: {len(text.split())}")

    print("\nIndexing...")
    chunks = chunk_text(text)
    vectors = embed_chunks(chunks)
    index = store_in_faiss(vectors)

    print(f"Chunks: {len(chunks)}")
    print("Vectors stored:", index.ntotal)

    while True:
        try:
            query = input("\nAsk a question (or type 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!.")
            break

        if not query:
            print("Please enter a question.")
            continue
        if query.lower() == "quit":
            print("  Bye!")
            break
        if len(query) > 500:
            print("  Please keep query under 500 characters.")
            continue

        results = search(query, index, chunks)
        context = build_context(results)
        prompt = build_prompt(context, query)
        answer = generate_answer(prompt)
        print("\n> ", end="")
        print_answer(answer)


if __name__ == "__main__":
    main()
