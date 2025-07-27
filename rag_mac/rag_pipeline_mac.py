import sys
import os
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from custom_bert import CustomBERTEmbedding

VOCAB_SIZE = 30522
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MAX_SEQ_LENGTH = 128

# Tokenizer
def simple_tokenizer(text, vocab_size=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH):
    tokens = text.lower().split()
    ids = [abs(hash(token)) % vocab_size for token in tokens]
    if len(ids) < max_seq_length:
        ids += [0] * (max_seq_length - len(ids))
    else:
        ids = ids[:max_seq_length]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# Data loading
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded = list(loaded.values())
        return loaded

def embed_text(model, text):
    input_ids = simple_tokenizer(text)
    with torch.no_grad():
        emb = model(input_ids)
        emb_vec = emb.mean(dim=1).squeeze().cpu().numpy()
    return emb_vec

def batch_embed_lectures(model, data, save_path):
    print(f"Embedding {len(data)} lectures. This may take a while...")
    embs = []
    meta = []
    for item in tqdm(data, desc="Embedding lectures"):
        lecture = item.get('lecture', '')
        embs.append(embed_text(model, lecture))
        # Store all relevant fields for retrieval
        meta.append({
            'question': item.get('question', ''),
            'choices': item.get('choices', []),
            'answer': item.get('answer', None),
            'lecture': lecture,
            'topic': item.get('topic', ''),
            'category': item.get('category', ''),
            'solution': item.get('solution', '')
        })
    embs = np.stack(embs)
    np.save(save_path, embs)
    # Save meta information for each embedding
    meta_path = save_path.replace('.npy', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(embs)} lecture embeddings to {save_path} and metadata to {meta_path}")

def load_embeddings(path):
    return np.load(path)

def retrieve(query_emb, data_embs, top_k=1):
    norms = np.linalg.norm(data_embs, axis=1) * np.linalg.norm(query_emb)
    sims = np.dot(data_embs, query_emb) / (norms + 1e-8)
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return top_indices

def gpt2_generate(prompt, max_length=128):
    # Load GPT-2 model and tokenizer (loads once, then caches)
    if not hasattr(gpt2_generate, "model"):
        gpt2_generate.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_generate.model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_generate.model.eval()
    tokenizer = gpt2_generate.tokenizer
    model = gpt2_generate.model
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_length=input_ids.shape[1]+max_length, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()

def evaluate_on_test_set(model, data_train, data_train_embs, data_test, save_csv_path=None):
    results = []
    correct_by_grade = {}
    total_by_grade = {}
    correct_by_subject = {}
    total_by_subject = {}
    for item in tqdm(data_test, desc="Evaluating test set"):        
        query = item.get('question', '')
        choices = item.get('choices', [])
        gt_index = item.get('answer', None)  # ground truth index (should be int)
        if gt_index is not None:
            try:
                gt_index = int(gt_index)
            except:
                gt_index = None
        grade = item.get('grade', 'Unknown')
        subject = item.get('subject', 'Unknown')
        query_emb = embed_text(model, query)
        top_idx = retrieve(query_emb, data_train_embs, top_k=1)[0]
        hit = data_train[top_idx]
        # Prompt instructs model to answer ONLY with the index
        prompt = (
            "You are a helpful science assistant. For each question, select the correct answer from the choices below. "
            "Respond ONLY with the index (0, 1) of the correct choice. Do not include any explanation or text, just the number.\n"
            f"Question: {query}\nChoices: {choices}\n"
            f"Lecture: {hit.get('lecture', '')}\n"
            f"Topic: {hit.get('topic', '')}\n"
            f"Category: {hit.get('category', '')}\n"
            f"Solution: {hit.get('solution', '')}\n"
            f"Answer: {hit.get('answer', '')}\n"
            "Your answer:"
        )
        try:
            pred_answer_raw = gpt2_generate(prompt)
            # Extract index from GPT-2 output (0 or 1 only)
            import re
            match = re.search(r'\b([01])\b', pred_answer_raw)
            pred_index = int(match.group(1)) if match else None
        except Exception as e:
            pred_index = None
        is_correct = (pred_index == gt_index)
        results.append({
            'question': query,
            'choices': choices,
            'ground_truth_index': gt_index,
            'predicted_index': pred_index,
            'grade': grade,
            'subject': subject,
            'is_correct': is_correct
        })
        # Grade stats
        correct_by_grade[grade] = correct_by_grade.get(grade, 0) + int(is_correct)
        total_by_grade[grade] = total_by_grade.get(grade, 0) + 1
        # Subject stats
        correct_by_subject[subject] = correct_by_subject.get(subject, 0) + int(is_correct)
        total_by_subject[subject] = total_by_subject.get(subject, 0) + 1
    # Print accuracy by grade
    print("\nAccuracy by Grade:")
    for grade in sorted(total_by_grade):
        acc = correct_by_grade[grade] / total_by_grade[grade]
        print(f"  Grade {grade}: {acc:.2%} ({correct_by_grade[grade]}/{total_by_grade[grade]})")
    # Print accuracy by subject
    print("\nAccuracy by Subject:")
    for subject in sorted(total_by_subject):
        acc = correct_by_subject[subject] / total_by_subject[subject]
        print(f"  Subject {subject}: {acc:.2%} ({correct_by_subject[subject]}/{total_by_subject[subject]})")
    # Save results
    if save_csv_path:
        df = pd.DataFrame(results)
        df.to_csv(save_csv_path, index=False)
        print(f"\nSaved detailed results to {save_csv_path}")
    return results, correct_by_grade, total_by_grade, correct_by_subject, total_by_subject

# --- Main RAG pipeline ---
def main():
    model = CustomBERTEmbedding(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LENGTH)
    model.load_state_dict(torch.load("models/custom_bert.pth", map_location=torch.device('cpu')))
    model.eval()
    data_train = load_data("data/train.json")
    emb_path = "data/lecture_embeddings.npy"
    meta_path = emb_path.replace('.npy', '_meta.json')
    if not os.path.exists(emb_path):
        batch_embed_lectures(model, data_train, emb_path)
    data_train_embs = load_embeddings(emb_path)
    # Load meta information for each embedding
    with open(meta_path, 'r', encoding='utf-8') as f:
        data_train_meta = json.load(f)
    # --- Test set evaluation ---
    test_path = "data/test.json"
    if os.path.exists(test_path):
        data_test = load_data(test_path)
        print("\nRunning evaluation on test set...")
        evaluate_on_test_set(
            model, data_train_meta, data_train_embs, data_test,
            save_csv_path="data/rag_test_results.csv"
        )
    else:
        print("Test set not found. Running in interactive mode.")
        while True:
            query = input("Enter your question (or 'exit'): ").strip()
            if not query or query.lower() == 'exit':
                break
            choices = input("Enter choices separated by '|||': ").strip().split('|||')
            query_emb = embed_text(model, query)
            top_idx = retrieve(query_emb, data_train_embs, top_k=1)[0]
            hit = data_train_meta[top_idx]
            prompt = (
                f"Question: {query}\nChoices: {choices}\n"
                f"Lecture: {hit.get('lecture', '')}\n"
                f"Topic: {hit.get('topic', '')}\n"
                f"Category: {hit.get('category', '')}\n"
                f"Solution: {hit.get('solution', '')}\n"
                f"Answer: {hit.get('answer', '')}\n"
                "Your answer:"
            )
            try:
                answer = gpt2_generate(prompt)
                print(f"\nBest Lecture: {hit.get('lecture', '')}")
                print(f"Topic: {hit.get('topic', '')}")
                print(f"Category: {hit.get('category', '')}")
                print(f"Solution: {hit.get('solution', '')}")
                print(f"Answer: {hit.get('answer', '')}")
                print(f"GPT-2 Answer: {answer}\n")
            except Exception as e:
                print(f"Error generating answer: {e}\n")

if __name__ == "__main__":
    main()
