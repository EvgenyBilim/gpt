

vocab_size = 50257    # стандартное для GPT2
block_size = 1024     # сколько токенов в одном куске текста
n_layer = 12  # 12
n_head = 12  # 12
n_embd = 768  # 768

dropout = 0.1

batch_size = 32  # 6
learning_rate = 3e-4
max_iters = 50_000
eval_interval = 1000
eval_iters = 10  # 200
log_interval = 10

train_file = "src/data/train2.txt"
