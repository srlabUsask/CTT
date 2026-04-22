import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Config, RobertaTokenizer


class Example(object):
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename, limit=1000):
    print(f"Reading examples from: {filename}")
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            js = json.loads(line.strip())
            code = " ".join(js["code_tokens"]).replace("\n", " ").strip()
            docstring = " ".join(js["docstring_tokens"]).replace("\n", " ").strip()
            examples.append(Example(idx=idx, source=code, target=docstring))
    print(f"  Loaded {len(examples)} examples")
    return examples


class InputFeatures(object):
    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    print(f"Converting {len(examples)} examples to features [{stage}]...")
    features = []
    for example in examples:
        source_tokens = tokenizer.tokenize(example.source)[: max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * len(source_ids)
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        target_tokens = tokenizer.tokenize(example.target if stage != "test" else "None")[: max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(InputFeatures(example.idx, source_ids, target_ids, source_mask, target_mask))
    print(f"  Created {len(features)} features")
    return features


class StudentT5(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff):
        super().__init__()
        config = T5Config.from_pretrained("Salesforce/codet5-base-multi-sum")
        config.d_model = d_model
        config.num_layers = num_layers
        config.num_heads = num_heads
        config.d_ff = d_ff
        self.model = T5ForConditionalGeneration(config)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss


def create_dataloader(filename, tokenizer, batch_size, max_source_length, max_target_length, stage, max_limit):
    examples = read_examples(filename, limit=max_limit)
    features = convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage)

    source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(source_ids, source_mask, target_ids, target_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(stage == "train"))


def fitness(candidate, train_loader, val_loader, device):
    print(f"  Evaluating candidate: {candidate}")
    model = StudentT5(**candidate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    print("  Training...")
    for batch in train_loader:
        input_ids, attention_mask, labels, _ = [b.to(device) for b in batch]
        loss = model(input_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("  Evaluating loss...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels, _ = [b.to(device) for b in batch]
            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"    Validation loss: {avg_loss:.4f}")
    return avg_loss


def search_algorithm(
    train_loader, val_loader, device, pop_size=50, generations=50, crossover_rate=0.7, mutation_rate=0.1
):
    print("Starting neural architecture search...")
    search_space = {
        "num_layers": (2, 12),
        "num_heads": (2, 12),
        "d_ff": (8, 3072),
    }

    def random_candidate():
        num_heads = random.choice([2, 4, 6, 8, 10, 12])
        head_dim = random.choice([4, 8, 16, 32, 64, 96, 128])
        d_model = num_heads * head_dim
        return {
            "d_model": d_model,
            "num_layers": random.randint(*search_space["num_layers"]),
            "num_heads": num_heads,
            "d_ff": random.randint(*search_space["d_ff"]),
        }

    def crossover(p1, p2):
        return {key: random.choice([p1[key], p2[key]]) for key in p1}

    def mutate(cand):
        if random.random() < mutation_rate:
            cand["num_layers"] = random.randint(*search_space["num_layers"])
        if random.random() < mutation_rate:
            cand["num_heads"] = random.randint(*search_space["num_heads"])
        if random.random() < mutation_rate:
            cand["d_ff"] = random.randint(*search_space["d_ff"])
        head_dim = random.choice([4, 8, 16, 32, 64, 96, 128])
        cand["d_model"] = cand["num_heads"] * head_dim
        return cand

    population = [random_candidate() for _ in range(pop_size)]

    for gen in range(generations):
        print(f"\n==== Generation {gen + 1} ====")
        scored = sorted(population, key=lambda c: fitness(c, train_loader, val_loader, device))
        best = scored[0]
        print(f"Generation {gen + 1} complete. Best architecture = {best}")

        next_gen = scored[:2]  # Elitism
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(scored[:4], 2)
            child = crossover(p1, p2) if random.random() < crossover_rate else p1.copy()
            child = mutate(child)
            next_gen.append(child)
        population = next_gen

    return scored[0]


def main():
    print("Starting NAS process for CodeT5 summarization...")
    train_file = "dataset/java/train.jsonl"
    valid_file = "dataset/java/valid.jsonl"
    batch_size = 8
    max_source_length = 256
    max_target_length = 128
    max_limit = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base-multi-sum")

    train_loader = create_dataloader(
        train_file, tokenizer, batch_size, max_source_length, max_target_length, "train", max_limit
    )
    val_loader = create_dataloader(
        valid_file, tokenizer, batch_size, max_source_length, max_target_length, "dev", max_limit
    )

    best_arch = search_algorithm(train_loader, val_loader, device)
    print("\n===== NAS Complete =====")
    print("Best architecture found:", best_arch)


if __name__ == "__main__":
    main()
