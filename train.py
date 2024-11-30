import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from tokenizer import CharacterTokenizer
from datasets import load_dataset
from model import PoetryGenerator

# 数据加载和预处理
def prepare_dataset(dataset, tokenizer, max_length=125, batch_size=64):
    def preprocess_batch(batch):
        texts = [item["text"] for item in batch]
        input_ids = tokenizer(
                        texts,
                        max_length=max_length,
                        add_special_tokens=True,
                        padding=True,
                        return_tensors="pt"
                    )
        return input_ids

    return DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=preprocess_batch)
        
# 保存模型
def save_model(model, path, is_parallel=False):
    if is_parallel:  # 如果是 DataParallel
        torch.save(model.module.state_dict(), path)
    else:  # 如果是普通模型或 DDP
        torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def lstm_weights_init(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


def train(machine_poet, dataloader, epochs=5, lr=0.001, weight_decay=1e-4, device='cpu', data_parallel=True, writer=None):
    machine_poet.model.apply(lstm_weights_init)
    criterion = nn.CrossEntropyLoss(ignore_index=machine_poet.tokenizer.pad_token_id)
    optimizer = optim.Adam(machine_poet.model.parameters(), lr=lr, weight_decay=weight_decay)
    if data_parallel:
        machine_poet.model = nn.DataParallel(machine_poet.model)  # 使用 DataParallel
    machine_poet.model = machine_poet.model.to(device)
    machine_poet.model.train()
    print("Start Training...")

    best_loss = float("inf")  # 用于保存最优模型
    for epoch in range(epochs):
        total_loss = 0
        trained_items = 0  # 重置计数器

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=120)
        for batch_idx, data in enumerate(pbar):
            inputs, targets = data[:,:-1].to(device), data[:,1:].to(device)
            optimizer.zero_grad()
            output, _ = machine_poet.model(inputs)
            loss = criterion(output, targets.reshape(-1))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) 
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            trained_items += 1

            # 更新进度条
            avg_batch_loss = total_loss / trained_items
            pbar.set_postfix_str(f"Loss: {avg_batch_loss:.4f}")

            # 每 10 个 batch 记录一次
            if batch_idx % 10 == 0:
                writer.add_scalar("Loss/train", avg_batch_loss, epoch * len(dataloader) + batch_idx)

        # 每个 epoch 记录一次平均损失
        avg_epoch_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train-Epoch", avg_epoch_loss, epoch)

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")

        # 保存最优模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_model(machine_poet.model, "model/best_poetry_lstm.bin", is_parallel=data_parallel)

        generated_texts = machine_poet.generate_poetry_by_start("江流天地外", 48)
        print(generated_texts)

        generated_texts = machine_poet.generate_acrostic("小牛小牛", 48)
        print(generated_texts)

    # 保存最终模型
    save_model(machine_poet.model, "model/poetry_lstm_final.bin", is_parallel=data_parallel)

if __name__ == "__main__":
    max_length = 125
    batch_size = 64

    embed_size = 128
    hidden_size = 256
    num_layers = 2

    dataset = load_dataset("FuryMartin/poetry_dataset")
    tokenizer = CharacterTokenizer.from_pretrained("vocab.json")

    dataloader = prepare_dataset(dataset, tokenizer, max_length=max_length, batch_size=batch_size)

    machine_poet = PoetryGenerator(embed_size, hidden_size, num_layers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TensorBoard
    writer = SummaryWriter("logs/")
    
    # 训练模型
    train(machine_poet, dataloader, epochs=50, lr=0.001, device=device, data_parallel=False, writer=writer)