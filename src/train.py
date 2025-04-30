from datetime import datetime
import os
import torch

from src.gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer


# Гиперпараметры
from src.config import block_size, n_layer, n_head, n_embd, dropout, train_file
from src.config import batch_size, learning_rate, max_iters, eval_interval, eval_iters, log_interval

# Девайс
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
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
