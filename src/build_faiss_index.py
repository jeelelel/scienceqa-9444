import os
import json
import numpy as np
import faiss

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EMBED_FIELD = 'lecture_embedding'  # You can change to question_embedding or others
EMBED_FILE = os.path.join(DATA_DIR, 'ScienceQA_train_embedded.json')
INDEX_FILE = os.path.join(DATA_DIR, 'faiss_index.bin')
IDMAP_FILE = os.path.join(DATA_DIR, 'faiss_idmap.json')


def build_faiss_index():
    with open(EMBED_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    vectors = []
    id_map = []
    for idx, item in enumerate(data):
        vec = item.get(EMBED_FIELD)
        if vec is not None:
            vectors.append(vec)
            id_map.append(idx)
    vectors = np.array(vectors).astype('float32')
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity (embeddings are normalized)
    index.add(vectors)
    faiss.write_index(index, INDEX_FILE)
    with open(IDMAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(id_map, f)
    print(f'FAISS index saved to {INDEX_FILE}, id mapping saved to {IDMAP_FILE}')


def main():
    build_faiss_index()


if __name__ == '__main__':
    main()
