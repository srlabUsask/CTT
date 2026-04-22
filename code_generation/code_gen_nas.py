import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Config,
    DataCollatorForLanguageModeling,
    logging
)

logging.set_verbosity_error()

class StudentCodeGen(nn.Module):
    def __init__(self, d_model, n_layer, n_head, n_inner, model_id):
        super().__init__()
        config = GPT2Config.from_pretrained(model_id)
        config.n_embd = d_model
        config.n_layer = n_layer
        config.n_head = n_head
        config.n_inner = n_inner
        config.vocab_size = config.vocab_size
        self.model = AutoModelForCausalLM.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss


def extract_prompt(code_text):
    pattern = r"(def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:[\s\S]*?(\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''))"
    match = re.search(pattern, code_text)
    return match.group(0) if match else None


def preprocess_data(tokenizer, split, limit, max_length):
    dataset = load_dataset("code_search_net", "python", split=f"{split}[:{limit}]")

    def preprocess(examples):
        texts = examples["func_code_string"]
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def fitness(candidate, tokenizer, train_data, val_data, device, model_id):
    print(f"Evaluating: {candidate}")
    model = StudentCodeGen(**candidate, model_id=model_id).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)

    model.train()
    for i in range(1):  # 1 epoch per candidate
        for batch in tqdm(train_data, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_data, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            total_loss += loss.item()
    avg_loss = total_loss / len(val_data)
    print(f"Val Loss: {avg_loss:.4f}")
    return avg_loss


def search_algorithm(train_data, val_data, tokenizer, device, model_id,
                     pop_size=50, generations=50, crossover_rate=0.7, mutation_rate=0.1):
    search_space = {
        "n_layer": (2, 12),
        "n_head": (2, 12),
        "n_inner": (8, 2048),
    }

    def random_candidate():
        n_head = random.choice([2, 4, 6, 8, 10, 12])
        head_dim = random.choice([4, 8, 16, 32, 64, 96, 128])
        d_model = n_head * head_dim
        return {
            "d_model": d_model,
            "n_layer": random.randint(*search_space["n_layer"]),
            "n_head": n_head,
            "n_inner": random.randint(*search_space["n_inner"]),
        }

    def crossover(p1, p2):
        return {k: random.choice([p1[k], p2[k]]) for k in p1}

    def mutate(cand):
        if random.random() < mutation_rate:
            cand["n_layer"] = random.randint(*search_space["n_layer"])
        if random.random() < mutation_rate:
            cand["n_head"] = random.randint(*search_space["n_head"])
        if random.random() < mutation_rate:
            cand["n_inner"] = random.randint(*search_space["n_inner"])
        head_dim = random.choice([4, 8, 16, 32, 64, 96, 128])
        cand["d_model"] = cand["n_head"] * head_dim
        return cand

    population = [random_candidate() for _ in range(pop_size)]

    for gen in range(generations):
        print(f"\n=== Generation {gen+1} ===")
        scored = sorted(
            population,
            key=lambda c: fitness(c, tokenizer, train_data, val_data, device, model_id)
        )
        print("Top candidate:", scored[0])

        next_gen = scored[:2]  # elitism
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(scored[:4], 2)
            if random.random() < crossover_rate:
                child = crossover(p1, p2)
            else:
                child = p1.copy()
            child = mutate(child)
            next_gen.append(child)
        population = next_gen

    return scored[0]


def main():
    model_id = "Salesforce/codegen-350M-mono"
    batch_size = 8
    max_length = 512
    train_limit = 5000
    valid_limit = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("Preprocessing...")
    train_dataset = preprocess_data(tokenizer, "train", train_limit, max_length)
    val_dataset = preprocess_data(tokenizer, "validation", valid_limit, max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    best_arch = search_algorithm(train_loader, val_loader, tokenizer, device, model_id)
    print("\n===== NAS Completed =====")
    print("Best architecture found:", best_arch)


if __name__ == "__main__":
    main()
