import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from custom_bert import CustomBERTEmbedding

# Dummy dataset for demonstration
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_length, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.labels = torch.randint(0, vocab_size, (num_samples, seq_length))
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    def __len__(self):
        return len(self.data)

# Hyperparameters
vocab_size = 30522
embed_dim = 256
num_heads = 8
num_layers = 6
max_seq_length = 128
batch_size = 16
epochs = 500
lr = 1e-3

# Model, dataset, optimizer
model = CustomBERTEmbedding(vocab_size, embed_dim, num_heads, num_layers, max_seq_length)
dataset = DummyDataset(vocab_size, max_seq_length, 100)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

max_epochs = 1000
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    for input_ids, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids)  # (batch, seq_len, vocab_size)
        logits = outputs.reshape(-1, vocab_size)  # (batch * seq_len, vocab_size)
        labels = labels.reshape(-1)  # (batch * seq_len)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Always save to absolute path for clarity
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_bert.pth'))
        torch.save(model.state_dict(), save_path)
        print(f"Model improved and saved as {save_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
            break
# Always save final model for clarity
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_bert.pth'))
torch.save(model.state_dict(), save_path)
print(f"Final model saved as {save_path}")
