import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use advanced BERT and optionally RoBERTa for comparison
BERT_MODEL_NAME = 'bert-large-uncased'
ROBERTA_MODEL_NAME = 'roberta-large'

# --- Embedding Functions ---
def embed_text(model, tokenizer, text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        emb_vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb_vec

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded = list(loaded.values())
        return loaded

def batch_embed_lectures(model, tokenizer, data, save_path):
    embs = []
    for item in tqdm(data, desc="Embedding lectures (BERT)"):
        concat_text = (
            f"Question: {item.get('question', '')}\n"
            f"Choices: {item.get('choices', [])}\n"
            f"Lecture: {item.get('lecture', '')}\n"
            f"Topic: {item.get('topic', '')}\n"
            f"Category: {item.get('category', '')}\n"
            f"Solution: {item.get('solution', '')}\n"
            f"Answer: {item.get('answer', '')}"
        )
        embs.append(embed_text(model, tokenizer, concat_text))
    embs = np.stack(embs)
    np.save(save_path, embs)
    print(f"Saved {len(embs)} BERT embeddings to {save_path}")

def load_embeddings(path):
    return np.load(path)

def evaluate_hybrid(model, tokenizer, data_train, bert_embs, tfidf_matrix, vectorizer, data_test, save_csv_path=None):
    results = []
    for item in tqdm(data_test, desc="Hybrid RAG Evaluation"):
        query = item.get('question', '')
        choices = item.get('choices', [])
        gt_index = item.get('answer', None)
        if gt_index is not None:
            try:
                gt_index = int(gt_index)
            except:
                gt_index = None
        grade = item.get('grade', 'Unknown')
        subject = item.get('subject', 'Unknown')
        # Hybrid retrieval
        query_emb = embed_text(model, tokenizer, query)
        bert_sims = np.dot(bert_embs, query_emb) / (np.linalg.norm(bert_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        tfidf_query = vectorizer.transform([query])
        tfidf_sims = cosine_similarity(tfidf_matrix, tfidf_query).flatten()
        hybrid_sims = 0.5 * bert_sims + 0.5 * tfidf_sims
        top_idx = np.argmax(hybrid_sims)
        hit = data_train[top_idx]
        choices_str = ', '.join(str(c) for c in choices)
        prompt = (
            "You are a helpful science assistant. For each question, select the correct answer from the choices below. "
            "Respond ONLY with the index (0, 1, 2, ...) of the correct choice. Do not include any explanation or text, just the number.\n"
            f"Question: {query}\nChoices: {choices_str}\n"
            f"Lecture: {hit.get('lecture', '')}\n"
            f"Topic: {hit.get('topic', '')}\n"
            f"Category: {hit.get('category', '')}\n"
            f"Solution: {hit.get('solution', '')}\n"
            f"Answer: {hit.get('answer', '')}\n"
            "Your answer:"
        )
        # Use GPT-2 for answer generation
        if not hasattr(evaluate_hybrid, "model"):
            evaluate_hybrid.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            evaluate_hybrid.model = GPT2LMHeadModel.from_pretrained("gpt2")
            evaluate_hybrid.model.eval()
        gpt2_tokenizer = evaluate_hybrid.tokenizer
        gpt2_model = evaluate_hybrid.model
        max_gpt2_len = 1024
        input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] > max_gpt2_len - 32:
            input_ids = input_ids[:, - (max_gpt2_len - 32):]
        with torch.no_grad():
            output = gpt2_model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 32,
                do_sample=False,
                pad_token_id=gpt2_tokenizer.eos_token_id
            )
        gen_start = input_ids.shape[1]
        gen_end = output.shape[1]
        if gen_start < gen_end:
            pred_answer_raw = gpt2_tokenizer.decode(output[0][gen_start:gen_end], skip_special_tokens=True).strip()
        else:
            pred_answer_raw = ""
        import re
        match = re.search(r'\b(\d+)\b', pred_answer_raw)
        pred_index = int(match.group(1)) if match else None
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
    if save_csv_path:
        df = pd.DataFrame(results)
        df.to_csv(save_csv_path, index=False)
        print(f"Saved hybrid RAG results to {save_csv_path}")
    return results

def main():
    # Load models
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
    # Optionally, load RoBERTa for comparison
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model = AutoModel.from_pretrained(ROBERTA_MODEL_NAME)
    # Load data
    data_train = load_data("data/train.json")
    data_test = load_data("data/test.json")
    # Prepare lecture texts
    emb_path = "../data/advanced_bert_lecture_embeddings.npy"
    if not os.path.exists(emb_path):
        batch_embed_lectures(bert_model, bert_tokenizer, data_train, emb_path)
    bert_embs = load_embeddings(emb_path)
    # TF-IDF setup
    lectures = [item.get('lecture', '') for item in data_train]
    vectorizer = TfidfVectorizer(max_features=4096)
    tfidf_matrix = vectorizer.fit_transform(lectures)
    # Evaluate hybrid pipeline
    results = evaluate_hybrid(bert_model, bert_tokenizer, data_train, bert_embs, tfidf_matrix, vectorizer, data_test, save_csv_path="../data/advanced_bert_hybrid_results.csv")
    print("Hybrid RAG evaluation complete.")

if __name__ == "__main__":
    main()
