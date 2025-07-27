import torch
from custom_bert import CustomBERTEmbedding

# Hyperparameters (should match training)
vocab_size = 30522
embed_dim = 256
num_heads = 8
num_layers = 6
max_seq_length = 128

# Load model
model = CustomBERTEmbedding(vocab_size, embed_dim, num_heads, num_layers, max_seq_length)
model.load_state_dict(torch.load("custom_bert.pth"))
model.eval()

# Dummy test input
input_ids = torch.randint(0, vocab_size, (1, max_seq_length))
with torch.no_grad():
    embeddings = model(input_ids)
    print("Embeddings shape:", embeddings.shape)
    print("Sample embedding vector:", embeddings[0, 0, :5])
