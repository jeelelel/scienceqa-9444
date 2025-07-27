import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_seq_length = max_seq_length
        self.output_layer = nn.Linear(embed_dim, vocab_size)  # Added for training

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        word_embeddings = self.embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = embeddings.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
        encoded = self.transformer_encoder(embeddings)
        encoded = encoded.permute(1, 0, 2)  # Back to (batch, seq_len, embed_dim)
        logits = self.output_layer(encoded)  # (batch, seq_len, vocab_size)
        return logits

# Example usage:
# model = CustomBERTEmbedding(vocab_size=30522, embed_dim=256, num_heads=8, num_layers=6, max_seq_length=128)
# input_ids = torch.randint(0, 30522, (32, 128))
# embeddings = model(input_ids)
