import random
from datasets import load_dataset
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import logging

logging.set_verbosity_error()


class CloneDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples["func1"])

    def __getitem__(self, idx):
        code1 = self.examples["func1"][idx] or "pass"
        code2 = self.examples["func2"][idx] or "pass"
        label = self.examples["label"][idx]
        inputs = self.tokenizer(
            code1, code2, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )
        return inputs["input_ids"].squeeze(), inputs["attention_mask"].squeeze(), torch.tensor(label, dtype=torch.long)


class StudentModel(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        config = RobertaConfig(
            vocab_size=50265,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=514,
        )
        self.encoder = RobertaModel(config)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits


def fitness(candidate, train_loader, val_loader, device):
    model = StudentModel(**candidate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, lbl = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(lbl.cpu().numpy())

    acc = accuracy_score(labels, preds)
    return -acc


def search_algorithm(
    train_loader, val_loader, device, pop_size=50, generations=50, crossover_rate=0.7, mutation_rate=0.1
):
    search_space = {
        "num_hidden_layers": (2, 12),
        "num_attention_heads": (2, 12),
        "intermediate_size": (8, 3072),
    }

    def random_candidate():
        valid_heads = [2, 4, 6, 8, 10, 12]
        valid_head_dims = [8, 16, 32, 64, 96, 128]

        num_attention_heads = random.choice(valid_heads)
        head_dim = random.choice(valid_head_dims)
        hidden_size = num_attention_heads * head_dim

        return {
            "num_hidden_layers": random.randint(2, 12),
            "num_attention_heads": num_attention_heads,
            "hidden_size": hidden_size,
            "intermediate_size": random.randint(256, 3072),
        }

    def crossover(p1, p2):
        child = {key: random.choice([p1[key], p2[key]]) for key in p1}
        child["hidden_size"] = 768
        return child

    def mutate(cand):
        if random.random() < mutation_rate:
            cand["num_hidden_layers"] = random.randint(*search_space["num_hidden_layers"])
        if random.random() < mutation_rate:
            cand["num_attention_heads"] = random.randint(*search_space["num_attention_heads"])
        if random.random() < mutation_rate:
            cand["intermediate_size"] = random.randint(*search_space["intermediate_size"])
        cand["hidden_size"] = 768
        return cand

    population = [random_candidate() for _ in range(pop_size)]

    for gen in range(generations):
        scored = sorted(population, key=lambda c: fitness(c, train_loader, val_loader, device))
        best = scored[0]
        print(f"Gen {gen+1}: Best accuracy = {-fitness(best, train_loader, val_loader, device):.4f}, Arch = {best}")

        next_gen = scored[:2]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(scored[:4], 2)
            child = crossover(p1, p2) if random.random() < crossover_rate else p1.copy()
            child = mutate(child)
            next_gen.append(child)
        population = next_gen

    return scored[0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")
    val_data = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")

    train_data = train_data.filter(lambda x: x["func1"] and x["func2"])
    val_data = val_data.filter(lambda x: x["func1"] and x["func2"])

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    tokenizer.model_max_length = 512

    train_dataset = CloneDataset(train_data, tokenizer)
    val_dataset = CloneDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    best_arch = search_algorithm(train_loader, val_loader, device)
    print("\nBest architecture:", best_arch)


if __name__ == "__main__":
    main()
