import argparse
import os
import torch

from datetime import datetime

from src.gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer

# from src.config import block_size, n_layer, n_head, n_embd, dropout, batch_size, max_iters
from src.config import learning_rate, eval_interval, eval_iters, log_interval, train_file


parser = argparse.ArgumentParser()
parser.add_argument('--n_layer', type=int, default=8)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_embd', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_iters', type=int, default=5_000)

args = parser.parse_args()

n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
batch_size = args.batch_size
max_iters = args.max_iters


# Девайс
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")


# Загружаем датасет
with open(train_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Токенизируем
tokenizer = SimpleTokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data))  # train/test split
train_data = data[:n]
val_data = data[n:]


# Получить батч данных
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# Оцениваем потери
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Модель
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
)
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Started at: {datetime.now()}")

# Тренировка
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    yb = yb.view(B*T)

    loss = torch.nn.functional.cross_entropy(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % log_interval == 0:
        print(f"iter {iter}: loss {loss.detach().item():.4f}")


# Сохраняем модель
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/mini_gpt2.pth")

print(f"Finished at: {datetime.now()}")
