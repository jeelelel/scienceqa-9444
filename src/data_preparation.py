import os
import json
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
HF_DATASET_NAME = 'derek-thomas/ScienceQA'
LOCAL_DATASET_DIR = os.path.join(DATA_DIR, 'ScienceQA')
GGUF_MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
GGUF_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'llama-2-7b-chat.Q4_K_M.gguf')


def check_and_download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Check if dataset already exists locally (check for at least one split file)
    expected_files = [
        os.path.join(DATA_DIR, f'ScienceQA_{split}.json') for split in ['train', 'validation', 'test']
    ]
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in expected_files):
        print('ScienceQA dataset already exists as filtered JSON files. No download needed.')
        return
    # If not, check for raw cache, else download
    if os.path.exists(LOCAL_DATASET_DIR) and len(os.listdir(LOCAL_DATASET_DIR)) > 0:
        print('ScienceQA raw dataset cache found. Will filter and save JSON files.')
    else:
        print('Downloading ScienceQA dataset from Hugging Face...')
        ds = load_dataset(HF_DATASET_NAME, cache_dir=LOCAL_DATASET_DIR)
        print('Download complete. Data cached at:', LOCAL_DATASET_DIR)
    # Always filter and save JSON files if not present
    ds = load_dataset(HF_DATASET_NAME, cache_dir=LOCAL_DATASET_DIR)
    for split in ['train', 'validation', 'test']:
        if split in ds:
            output_path = os.path.join(DATA_DIR, f'ScienceQA_{split}.json')
            extract_and_save_text_only(ds, split, output_path)


def extract_and_save_text_only(ds, split_name, output_path):
    text_fields = [
        'question', 'choices', 'lecture', 'explanation', 'answer', 'hint', 'topic', 'grade', 'subject', 'context', 'support', 'type', 'id'
    ]
    text_data = []
    for item in ds[split_name]:
        filtered = {k: v for k, v in item.items() if k in text_fields}
        text_data.append(filtered)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=2)
    print(f'Text data for {split_name} saved to {output_path}')


def prepare_text_only_data():
    ds = load_dataset(HF_DATASET_NAME, cache_dir=LOCAL_DATASET_DIR)
    for split in ['train', 'validation', 'test']:
        if split in ds:
            output_path = os.path.join(DATA_DIR, f'ScienceQA_{split}.json')
            extract_and_save_text_only(ds, split, output_path)


def check_and_download_gguf():
    if os.path.exists(GGUF_MODEL_PATH) and os.path.getsize(GGUF_MODEL_PATH) > 0:
        print(f'GGUF model already exists at {GGUF_MODEL_PATH}. No download needed.')
        return
    print(f'Downloading GGUF model from {GGUF_MODEL_URL} ...')
    import requests
    with requests.get(GGUF_MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(GGUF_MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f'Download complete. Model saved at {GGUF_MODEL_PATH}')


if __name__ == '__main__':
    check_and_download_gguf()
    check_and_download_dataset()
    prepare_text_only_data()
