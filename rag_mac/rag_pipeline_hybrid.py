import sys
import os
import torch
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from custom_bert import CustomBERTEmbedding

VOCAB_SIZE = 30522
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MAX_SEQ_LENGTH = 128

# Tokenizer (same as BERT)
def simple_tokenizer(text, vocab_size=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH):
    tokens = text.lower().split()
    ids = [abs(hash(token)) % vocab_size for token in tokens]
    if len(ids) < max_seq_length:
        ids += [0] * (max_seq_length - len(ids))
    else:
        ids = ids[:max_seq_length]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def embed_text(model, text):
    input_ids = simple_tokenizer(text)
    with torch.no_grad():
        emb = model(input_ids)
        emb_vec = emb.mean(dim=1).squeeze().cpu().numpy()
    return emb_vec

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
        if isinstance(loaded, dict):
            loaded = list(loaded.values())
        return loaded

def batch_embed_lectures(model, data, save_path):
    embs = []
    for item in tqdm(data, desc="Embedding lectures (BERT)"):
        # Concatenate all relevant fields for embedding
        concat_text = (
            f"Question: {item.get('question', '')}\n"
            f"Choices: {item.get('choices', [])}\n"
            f"Lecture: {item.get('lecture', '')}\n"
            f"Topic: {item.get('topic', '')}\n"
            f"Category: {item.get('category', '')}\n"
            f"Solution: {item.get('solution', '')}\n"
            f"Answer: {item.get('answer', '')}"
        )
        embs.append(embed_text(model, concat_text))
    embs = np.stack(embs)
    np.save(save_path, embs)
    print(f"Saved {len(embs)} BERT embeddings to {save_path}")

def evaluate_hybrid(model, data_train, bert_embs, tfidf_matrix, vectorizer, data_test, save_csv_path=None):
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
        query_emb = embed_text(model, query)
        bert_sims = np.dot(bert_embs, query_emb) / (np.linalg.norm(bert_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        tfidf_query = vectorizer.transform([query])
        tfidf_sims = cosine_similarity(tfidf_matrix, tfidf_query).flatten()
        hybrid_sims = 0.5 * bert_sims + 0.5 * tfidf_sims
        top_idx = np.argmax(hybrid_sims)
        hit = data_train[top_idx]
        # Prompt engineering
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
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        if not hasattr(evaluate_hybrid, "model"):
            evaluate_hybrid.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            evaluate_hybrid.model = GPT2LMHeadModel.from_pretrained("gpt2")
            evaluate_hybrid.model.eval()
        tokenizer = evaluate_hybrid.tokenizer
        gpt2_model = evaluate_hybrid.model
        # Truncate prompt if too long for GPT-2
        max_gpt2_len = 1024
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] > max_gpt2_len - 32:
            input_ids = input_ids[:, - (max_gpt2_len - 32):]  # take last tokens to keep answer context
        with torch.no_grad():
            output = gpt2_model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_start = input_ids.shape[1]
        gen_end = output.shape[1]
        if gen_start < gen_end:
            pred_answer_raw = tokenizer.decode(output[0][gen_start:gen_end], skip_special_tokens=True).strip()
        else:
            pred_answer_raw = ""
        import re
        match = re.search(r'\b([01])\b', pred_answer_raw)
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

def test_single_question(model, data_train, bert_embs, tfidf_matrix, vectorizer, test_item):
    """
    Test the hybrid RAG pipeline on a single question and print the result.
    """
    query = test_item.get('question', '')
    choices = test_item.get('choices', [])
    gt_index = test_item.get('answer', None)
    if gt_index is not None:
        try:
            gt_index = int(gt_index)
        except:
            gt_index = None
    grade = test_item.get('grade', 'Unknown')
    subject = test_item.get('subject', 'Unknown')
    # Hybrid retrieval
    query_emb = embed_text(model, query)
    bert_sims = np.dot(bert_embs, query_emb) / (np.linalg.norm(bert_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    tfidf_query = vectorizer.transform([query])
    tfidf_sims = cosine_similarity(tfidf_matrix, tfidf_query).flatten()
    hybrid_sims = 0.5 * bert_sims + 0.5 * tfidf_sims
    top_idx = np.argmax(hybrid_sims)
    hit = data_train[top_idx]
    choices_str = ', '.join(str(c) for c in choices)
    prompt = (
        "You are a helpful science assistant. For each question, select the correct answer from the choices below. "
        "Respond ONLY with the index (0, 1) of the correct choice. Do not include any explanation or text, just the number.\n"
        f"Question: {query}\nChoices: {choices_str}\n"
        f"Lecture: {hit.get('lecture', '')}\n"
        f"Topic: {hit.get('topic', '')}\n"
        f"Category: {hit.get('category', '')}\n"
        f"Solution: {hit.get('solution', '')}\n"
        f"Answer: {hit.get('answer', '')}\n"
        "Your answer:"
    )
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    if not hasattr(test_single_question, "model"):
        test_single_question.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        test_single_question.model = GPT2LMHeadModel.from_pretrained("gpt2")
        test_single_question.model.eval()
    tokenizer = test_single_question.tokenizer
    gpt2_model = test_single_question.model
    max_gpt2_len = 1024
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if input_ids.shape[1] > max_gpt2_len - 32:
        input_ids = input_ids[:, - (max_gpt2_len - 32):]
    with torch.no_grad():
        output = gpt2_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_start = input_ids.shape[1]
    gen_end = output.shape[1]
    if gen_start < gen_end:
        pred_answer_raw = tokenizer.decode(output[0][gen_start:gen_end], skip_special_tokens=True).strip()
    else:
        pred_answer_raw = ""
    import re
    match = re.search(r'\b([01])\b', pred_answer_raw)
    pred_index = int(match.group(1)) if match else None
    is_correct = (pred_index == gt_index)
    print("Question:", query)
    print("Choices:", choices)
    print("Ground Truth Index:", gt_index)
    print("Predicted Index:", pred_index)
    print("Raw Model Output:", pred_answer_raw)
    print("Is Correct:", is_correct)
    print("Grade:", grade, "Subject:", subject)
    return {
        'question': query,
        'choices': choices,
        'ground_truth_index': gt_index,
        'predicted_index': pred_index,
        'grade': grade,
        'subject': subject,
        'is_correct': is_correct,
        'raw_output': pred_answer_raw
    }

def main():
    # Load model and data
    model = CustomBERTEmbedding(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LENGTH)
    model.load_state_dict(torch.load("models/custom_bert.pth", map_location=torch.device('cpu')))
    model.eval()
    data_train = load_data("data/train.json")
    data_test = load_data("data/test.json")
    # Filter training data to only items with valid lectures
    filtered_train = [item for item in data_train if isinstance(item.get('lecture', ''), str) and item.get('lecture', '').strip()]
    # Use filtered_train for both BERT and TF-IDF
    lectures = [item.get('lecture', '') for item in filtered_train]
    bert_emb_path = "data/lecture_embeddings.npy"
    if not os.path.exists(bert_emb_path):
        batch_embed_lectures(model, filtered_train, bert_emb_path)
    bert_embs = np.load(bert_emb_path)
    tfidf_path = "data/lecture_tfidf.npz"
    vocab_path = "data/tfidf_vocab.json"
    if not os.path.exists(tfidf_path):
        vectorizer = TfidfVectorizer(max_features=4096)
        tfidf_matrix = vectorizer.fit_transform(lectures)
        from scipy import sparse
        sparse.save_npz(tfidf_path, tfidf_matrix)
        vocab_serializable = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
        if not vocab_serializable:
            print("Warning: TF-IDF vocabulary is empty. Aborting save.")
        else:
            with open(vocab_path, "w") as f:
                json.dump(vocab_serializable, f)
    else:
        from scipy import sparse
        tfidf_matrix = sparse.load_npz(tfidf_path)
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        if not vocab:
            print("Warning: Loaded TF-IDF vocabulary is empty. Aborting vectorizer creation.")
            vectorizer = None
        else:
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            vectorizer.fit(lectures)  # Fit the vectorizer so it can transform queries
    # Evaluate hybrid model only if TF-IDF matrix and vectorizer are valid
    if tfidf_matrix is not None and vectorizer is not None and bert_embs is not None:
        evaluate_hybrid(
            model, filtered_train, bert_embs, tfidf_matrix, vectorizer, data_test,
            save_csv_path="data/rag_hybrid_results.csv"
        )
    else:
        print("TF-IDF matrix, vectorizer, or BERT embeddings are invalid. Skipping hybrid evaluation.")

if __name__ == "__main__":
    main()
    # Example usage for testing a single question
    # model = CustomBERTEmbedding(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LENGTH)
    # model.load_state_dict(torch.load("models/custom_bert.pth", map_location=torch.device('cpu')))
    # model.eval()
    # data_train = load_data("data/train.json")
    # data_test = load_data("data/test.json")
    # filtered_train = [item for item in data_train if isinstance(item.get('lecture', ''), str) and item.get('lecture', '').strip()]
    # lectures = [item.get('lecture', '') for item in filtered_train]
    # bert_embs = np.load("data/lecture_embeddings.npy")
    # from scipy import sparse
    # tfidf_matrix = sparse.load_npz("data/lecture_tfidf.npz")
    # with open("data/tfidf_vocab.json", "r") as f:
    #     vocab = json.load(f)
    # vectorizer = TfidfVectorizer(vocabulary=vocab)
    # vectorizer.fit(lectures)
    # test_single_question(model, filtered_train, bert_embs, tfidf_matrix, vectorizer, data_test[0])
