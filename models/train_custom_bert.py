import torch
import torch.nn as nn
import torch.optim as optim
from custom_bert import CustomBERTEmbedding
import json
from tqdm import tqdm

# Hyperparameters
vocab_size = 30522
embed_dim = 256
num_heads = 8
num_layers = 6
max_seq_length = 128
batch_size = 16
epochs = 500  # Increased epochs
lr = 1e-3

def simple_tokenizer(text, vocab_size=30522, max_seq_length=128):
    tokens = text.lower().split()
    ids = [abs(hash(token)) % vocab_size for token in tokens]
    if len(ids) < max_seq_length:
        ids += [0] * (max_seq_length - len(ids))
    else:
        ids = ids[:max_seq_length]
    return torch.tensor(ids, dtype=torch.long)

class ScienceQADataset(torch.utils.data.Dataset):
    def __init__(self, json_paths, vocab_size, max_seq_length):
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        self.data = []
        for path in json_paths:
            with open(path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # If loaded is a dict (as in your data), convert to list of dicts
                if isinstance(loaded, dict):
                    loaded = list(loaded.values())
                for item in loaded:
                    if isinstance(item, dict):
                        self.data.append(item)
                    else:
                        print(f"Warning: Skipping non-dict item in {path}: {item}")
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get('question', '')
        input_ids = simple_tokenizer(question, self.vocab_size, self.max_seq_length)
        # For demonstration, use input_ids as labels (autoencoding)
        labels = input_ids.clone()
        return input_ids, labels
    def __len__(self):
        return len(self.data)

# Model, dataset, optimizer
model = CustomBERTEmbedding(vocab_size, embed_dim, num_heads, num_layers, max_seq_length)
dataset = ScienceQADataset([
    'data/train.json',
    'data/test.json'
], vocab_size, max_seq_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# Early stopping parameters
patience = 5
best_loss = float('inf')
no_improve = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_ids, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        logits = model(input_ids)  # (batch, seq_len, vocab_size)
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
        torch.save(model.state_dict(), "custom_bert.pth")
        print("Model improved and saved.")
    else:
        no_improve += 1
        print(f"No improvement for {no_improve} epoch(s).")
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}.")
        break

# Save model (in case early stopping didn't trigger)
if no_improve < patience:
    torch.save(model.state_dict(), "custom_bert.pth")
    print("Model saved as custom_bert.pth")
