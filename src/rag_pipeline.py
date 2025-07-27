import os
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from models.custom_bert import CustomBERTEmbedding

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


VOCAB_SIZE = 30522
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MAX_SEQ_LENGTH = 128

def simple_tokenizer(text, vocab_size=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH):
    # Simple whitespace tokenizer, maps words to IDs by hash (for demo purposes)
    tokens = text.lower().split()
    ids = [abs(hash(token)) % vocab_size for token in tokens]
    # Pad or truncate
    if len(ids) < max_seq_length:
        ids += [0] * (max_seq_length - len(ids))
    else:
        ids = ids[:max_seq_length]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, seq_len)

def load_resources():
    index = faiss.read_index(INDEX_FILE)
    with open(IDMAP_FILE, 'r', encoding='utf-8') as f:
        id_map = json.load(f)
    with open(EMBED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model = CustomBERTEmbedding(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LENGTH)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../models/custom_bert.pth')))
    model.eval()
    return index, id_map, data, model


def rag_qa(query, choices, index, id_map, data, model, top_k=TOP_K):
    # 1. Convert question to embedding using custom model
    input_ids = simple_tokenizer(query)
    with torch.no_grad():
        emb = model(input_ids)  # (1, seq_len, embed_dim)
        q_emb = emb.mean(dim=1).cpu().numpy().astype('float32')  # Mean pooling
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
    # 4. Return answer as index only (0 or 1)
    answer_idx = hit.get('answer', '')
    if isinstance(answer_idx, int) or (isinstance(answer_idx, str) and answer_idx.isdigit()):
        idx = int(answer_idx)
        answer = str(idx)
    else:
        answer = str(answer_idx)
    return answer, lecture, solution


def test_rag_pipeline(test_file):
    index, id_map, data, model = load_resources()
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        if isinstance(test_data, dict):
            test_data = list(test_data.values())
    correct = 0
    total = 0
    for i, item in enumerate(test_data):
        query = item.get('question', '')
        choices = item.get('choices', [])
        gt = item.get('answer', '')
        answer, lecture, solution = rag_qa(query, choices, index, id_map, data, model)
        # Compare answer index to ground truth
        try:
            gt_idx = int(gt)
            pred_idx = int(answer.split(':')[0]) if ':' in answer else None
            is_correct = (pred_idx == gt_idx)
        except:
            is_correct = (str(answer).strip().lower() == str(gt).strip().lower())
        correct += int(is_correct)
        total += 1
        if i < 5:
            print(f"Sample {i+1}")
            print("Question:", query)
            print("Choices:", choices)
            print("Ground Truth Index:", gt)
            print("Predicted Answer:", answer)
            print("Is Correct:", is_correct)
            print("Lecture:", lecture)
            print("Solution:", solution)
            print("---")
    print(f"\nRAG Pipeline Test Accuracy: {correct}/{total} = {correct/total:.2%}")

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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_file = os.path.join(DATA_DIR, 'test.json')
        test_rag_pipeline(test_file)
    else:
        main()
