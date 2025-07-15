import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EMBED_FIELD = 'lecture_embedding'  # Field used for retrieval
EMBED_FILE = os.path.join(DATA_DIR, 'ScienceQA_train_embedded.json')
INDEX_FILE = os.path.join(DATA_DIR, 'faiss_index.bin')
IDMAP_FILE = os.path.join(DATA_DIR, 'faiss_idmap.json')
# MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_NAME = 'data/finetuned-embedding'
TOP_K = 3

# Load your local GGUF model (update the path to your model file)
llm = Llama(model_path="./llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048)

def local_llm_generate(prompt):
    output = llm(prompt, max_tokens=256, stop=["\n"])
    return output["choices"][0]["text"].strip()


def load_resources():
    index = faiss.read_index(INDEX_FILE)
    with open(IDMAP_FILE, 'r', encoding='utf-8') as f:
        id_map = json.load(f)
    with open(EMBED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, id_map, data, model


def rag_qa(query, choices, index, id_map, data, model, top_k=TOP_K):
    # 1. Convert question to embedding
    q_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    # 2. FAISS retrieval
    D, I = index.search(q_emb, top_k)
    hits = [data[id_map[i]] for i in I[0]]
    hit = hits[0]
    lecture = hit.get('lecture', '')
    # 3. Construct structured prompt
    prompt = (
        f"Question:\n{query}\n\n"
        f"Choices:\n" + "\n".join([f"{i}. {c}" for i, c in enumerate(choices)]) + "\n\n"
        f"Lecture:\n{lecture}\n\n"
        "Please reason step by step based on the Lecture and provide the final answer. Output format:\n"
        "Solution:\n<Detailed reasoning>\nAnswer:\n<Final answer>"
    )
    solution = local_llm_generate(prompt)
    # 4. Return answer as index + content
    answer_idx = hit.get('answer', '')
    if isinstance(answer_idx, int) or (isinstance(answer_idx, str) and answer_idx.isdigit()):
        idx = int(answer_idx)
        if 0 <= idx < len(choices):
            answer = f"{idx}: {choices[idx]}"
        else:
            answer = str(answer_idx)
    else:
        answer = str(answer_idx)
    return answer, lecture, solution


def main():
    index, id_map, data, model = load_resources()
    while True:
        print("Please enter your question (type 'exit' to quit):")
        query = input("Question: ").strip()
        if query.lower() == 'exit':
            break
        print("Please enter the choices, separated by '|||':")
        choices = [c.strip() for c in input("Choices: ").strip().split('|||')]
        answer, lecture, solution = rag_qa(query, choices, index, id_map, data, model)
        print("\n==== RAG QA Result ====\n")
        print("Answer:", answer)
        print("Lecture:", lecture)
        print("Solution:", solution)
        print("\n======================\n")

if __name__ == '__main__':
    main()
