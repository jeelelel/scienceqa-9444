import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json
from transformers import BertTokenizer, BertModel

# Modern BERT model (can be changed to any from HuggingFace)
BERT_MODEL_NAME = 'bert-base-uncased'

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def batch_embed_texts(model, tokenizer, texts, save_path):
    print(f"Embedding {len(texts)} texts with {BERT_MODEL_NAME}. This may take a while...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    embs = []
    for text in tqdm(texts, desc="Embedding texts"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embs.append(emb)
    embs = np.stack(embs)
    np.save(save_path, embs)
    print(f"Saved {len(embs)} embeddings to {save_path}")

def load_embeddings(path):
    return np.load(path)

def prompt_engineer(question, context):
    # Simple prompt engineering: concatenate question and context
    return f"Question: {question}\nContext: {context}"

def retrieve(query_emb, data_embs, top_k=1):
    sims = np.dot(data_embs, query_emb) / (np.linalg.norm(data_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return top_indices

def main():
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME)
    # Load data
    data_train = load_data("../data/rain.json")
    data_test = load_data("../data/test.json")
    # Prepare lecture texts
    lecture_texts = [item.get('lecture', '') for item in data_train]
    emb_path = "../data/modern_bert_lecture_embeddings.npy"
    if not os.path.exists(emb_path):
        batch_embed_texts(model, tokenizer, lecture_texts, emb_path)
    data_train_embs = load_embeddings(emb_path)
    # Example: retrieve for a test question
    test_question = data_test[0]['question']
    # Prompt engineering: combine question and context (empty context for demo)
    prompt = prompt_engineer(test_question, "")
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items()})
        query_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    top_idx = retrieve(query_emb, data_train_embs, top_k=1)[0]
    print(f"Test question: {test_question}")
    print(f"Top retrieved lecture index: {top_idx}")
    print(f"Top retrieved lecture: {data_train[top_idx].get('lecture', '')}")

if __name__ == "__main__":
    main()
