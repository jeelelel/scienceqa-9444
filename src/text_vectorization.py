import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_NAME = 'all-MiniLM-L6-v2'  # You can change this to another local model
EMBED_FIELDS = ['question', 'lecture', 'explanation']


def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def process_and_save_embeddings(input_json, output_json, model):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in tqdm(data, desc=f'Embedding {os.path.basename(input_json)}'):
        for field in EMBED_FIELDS:
            text = item.get(field, '')
            if text:
                item[field + '_embedding'] = embed_texts([text], model)[0]
            else:
                item[field + '_embedding'] = None
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'Embedding data saved to {output_json}')


def main():
    model = SentenceTransformer(MODEL_NAME)
    for split in ['train', 'validation', 'test']:
        input_json = os.path.join(DATA_DIR, f'ScienceQA_{split}.json')
        output_json = os.path.join(DATA_DIR, f'ScienceQA_{split}_embedded.json')
        if os.path.exists(input_json):
            process_and_save_embeddings(input_json, output_json, model)


if __name__ == '__main__':
    main()
