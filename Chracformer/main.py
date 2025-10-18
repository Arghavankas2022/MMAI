#Script for a Medical character based transformer aka characformer
import torch
import torch.nn as nn
from torch.nn import functional as F

context_length = 6 #How many characters the model looks to predict the next one
batch_size = 8
torch.manual_seed(42)

with open("medical_terms.txt", "r") as f:
    text = f.read()

charachters = sorted(list(set(text)))
vocab_size = len(charachters)

# character level tokenizer: assigning an integer to each character
st_to_int = {ch: i for i, ch in enumerate(charachters)}
int_to_st = {i: ch for i, ch in enumerate(charachters)}
encode = lambda s: [st_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_st[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
test_data = data[n:]


# function for creating small batches of data
def make_batches(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])
    return x, y


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, n_heads=1): #vocab size: number of tokens in vocabulary
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1000, d_model)  # positional embeddings
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model) #Layer normalization
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        # x: (batch, seq_len)
        B, T = x.shape
        # Token + positional embeddings
        tok_emb = self.embed(x)  # (B, T, d_model)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.pos_embed(pos)  # (T, d_model)
        h = tok_emb + pos_emb  # (B, T, d_model)

        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)

        # Self-attention with residual connection
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.ln1(h + attn_out)

        # Feedforward with residual connection
        h = self.ln2(h + self.ffn(h))

        # Project to vocabulary
        logits = self.lm_head(h)  # (B, T, vocab_size)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last context_length tokens
            idx_cond = idx[:, -context_length:]
            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Initialize model with actual vocab size
model = MiniTransformer(vocab_size=vocab_size, d_model=64, n_heads=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# Training loop
def train(epochs=300, eval_interval=100):
    for epoch in range(epochs):
        model.train()
        xb, yb = make_batches("train")

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = make_batches("test")
                _, val_loss = model(x_val, y_val)
            print(f"Epoch {epoch}: train loss {loss.item():.4f}, val loss {val_loss.item():.4f}")
            model.train()

print("Training...")
train(epochs=300, eval_interval=100)

# Generate text
print("Generating text...")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long)  # Start with single token
generated = model.generate(context, max_new_tokens=1000)
print(decode(generated[0].tolist()))