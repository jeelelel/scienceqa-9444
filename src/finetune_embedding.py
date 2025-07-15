import os
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'ScienceQA_train.json')
OUTPUT_DIR = os.path.join(DATA_DIR, 'finetuned-embedding')

# 1. Load ScienceQA train data and construct InputExamples
with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

train_examples = []
for item in data:
    question = item.get('question', '')
    lecture = item.get('lecture', '')
    if question and lecture:
        train_examples.append(InputExample(texts=[question, lecture]))

# 2. Prepare DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 3. Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Define loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# 5. Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path=OUTPUT_DIR
)

print(f'Finetuned embedding model saved to {OUTPUT_DIR}')
