import torch
from src.gpt import GPT, GPTConfig
from src.tokenizer import SimpleTokenizer

from src.config import vocab_size, block_size, n_layer, n_head, n_embd, dropout, train_file

device = 'mps' if torch.backends.mps.is_available() else 'cpu'


# Загружаем данные для токенизатора
with open(train_file, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = SimpleTokenizer(text)

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
)

model = GPT(config)
model.load_state_dict(torch.load('src/checkpoints/mini_gpt2.pth', map_location=device))
model = model.to(device)
model.eval()


def generate(idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# Стартовый текст
# context = "ROMEO:"
context = "Хочет ли Австрия войны?"
idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long).to(device)

out = generate(idx, max_new_tokens=500)[0].tolist()
print(tokenizer.decode(out))
