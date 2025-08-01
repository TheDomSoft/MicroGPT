import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------
# Hyper-parameters (toy values)
# -----------------------------
batch_size   = 16     # how many sequences per batch
block_size   = 64     # context length (max 512 in this demo)
d_model      = 128    # embedding dimension
n_head       = 4      # attention heads
n_layer      = 2      # transformer layers
learning_rate = 1e-3
max_iters    = 3000   # training steps
eval_interval= 100
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------
# 1. Grab a tiny text corpus
# -----------------------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
import requests, os, hashlib, tempfile
cache = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode()).hexdigest())
if not os.path.isfile(cache):
    open(cache, 'w', encoding='utf-8').write(requests.get(url).text)
text = open(cache, encoding='utf-8').read()

# -----------------------------
# 2. Character-level tokenizer
# -----------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

# -----------------------------
# 3. Train / val split
# -----------------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------
# 4. Single-head causal attention
# -----------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)      # (B,T,hs)
        q = self.query(x)    # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * (C**-0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)    # (B,T,hs)
        return wei @ v        # (B,T,hs)

# -----------------------------
# 5. Multi-head attention block
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

# -----------------------------
# 6. Feed-forward block
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )

    def forward(self, x): return self.net(x)

# -----------------------------
# 7. Transformer block
# -----------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa  = MultiHeadAttention(n_head, d_model//n_head)
        self.ffw = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

# -----------------------------
# 8. Full model
# -----------------------------
class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(block_size, d_model)
        self.blocks    = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f      = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_emb(idx)                  # (B,T,d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=device))
        x = tok_emb + pos_emb                          # (B,T,d_model)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                       # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]            # crop context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------
# 9. Training loop
# -----------------------------
model = MicroGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------
# 10. Generate after training
# -----------------------------
context = torch.zeros((1,1), dtype=torch.long, device=device)  # start token 0
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))