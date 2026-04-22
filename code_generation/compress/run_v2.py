import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    default_data_collator,
)
from torch.optim import AdamW
import evaluate
from codebleu import calc_codebleu
from tqdm import tqdm
from quanto import quantize, qint8
import os
import numpy as np
from torch.utils.data import DataLoader
import random
from codecarbon import EmissionsTracker
import pandas as pd


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()


seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
GRAD_ACCUMULATION_STEPS = 2
OUTPUT_DIR = "compress-qwen-coder-1_5B-instruct-code"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
LAST_MODEL_DIR = os.path.join(OUTPUT_DIR, "last_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_decay = 0.05
adam_epsilon = 1e-8
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
teacher_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
teacher_model_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
eval_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
es = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")


tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)

special_tokens_dict = {"additional_special_tokens": ["[DONE]"]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens.")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_path, use_safetensors=True, trust_remote_code=True, device_map="auto"
)

teacher_model.resize_token_embeddings(len(tokenizer))
student_config = AutoConfig.from_pretrained(teacher_model_path, trust_remote_code=True)

student_config.num_attention_heads = 8
student_config.num_hidden_layers = 16
student_config.intermediate_size = 1024
student_config.hidden_size = 512

print(student_config)

student_model = AutoModelForCausalLM.from_config(student_config)
student_model.resize_token_embeddings(len(tokenizer))

quantize(student_model, weights=int8)

total_params = sum(p.numel() for p in student_model.parameters())

print(f"{total_params:,} total parameters.")
print(f"{total_params*4/1e6} MB model size")

student_model.to(device)


def extract_function_signature(code):
    import re

    lines = code.strip().split("\n")
    imports_and_signature = []

    for line in lines:
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports_and_signature.append(line)

    for line in lines:
        line = line.strip()
        if line.startswith("def "):
            if line.endswith(":"):
                imports_and_signature.append(line)
                break
            else:
                func_sig = line
                for next_line in lines[lines.index(line) + 1 :]:
                    func_sig += " " + next_line.strip()
                    if next_line.strip().endswith(":"):
                        imports_and_signature.append(func_sig)
                        break
                break

    return "\n".join(imports_and_signature)


def preprocess_function(examples):
    prompts = []
    for p, t in zip(examples["prompt"], examples["test_list"]):
        test_list_str = "\n".join(t)
        prompt = (
            f"You are an expert Python programmer, and here is your task: {p}\n"
            f"Your code should pass these tests and be syntactically correct:\n\n"
            f"{test_list_str}\n[BEGIN]\n"
        )
        prompts.append(prompt)

    codes = [c + "\n[DONE]" for c in examples["code"]]
    full_texts = [p + c for p, c in zip(prompts, codes)]

    model_inputs = tokenizer(
        full_texts,
        max_length=1024,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )

    prompt_enc = tokenizer(
        prompts,
        max_length=1024,
        truncation=True,
        add_special_tokens=False,
    )

    code_enc = tokenizer(
        codes,
        max_length=1024,
        truncation=True,
        add_special_tokens=False,
    )

    labels = []
    for i in range(len(model_inputs["input_ids"])):
        input_ids = model_inputs["input_ids"][i]
        attn = model_inputs["attention_mask"][i]
        prompt_len = len(prompt_enc["input_ids"][i])
        code_len = len(code_enc["input_ids"][i])

        try:
            first_nonpad = attn.index(1)
        except ValueError:
            first_nonpad = 0

        code_start = first_nonpad + prompt_len
        code_end = code_start + code_len

        lab = [-100] * len(input_ids)
        L = len(lab)
        s = min(max(code_start, 0), L)
        e = min(max(code_end, 0), L)

        actual_prompt_len = 0
        for j in range(len(input_ids)):
            if input_ids[j] == prompt_enc["input_ids"][i][j]:
                actual_prompt_len += 1
            else:
                break

        code_start = actual_prompt_len
        code_end = actual_prompt_len + code_len

        lab[s:e] = input_ids[s:e]
        labels.append(lab)

    model_inputs["labels"] = labels
    return model_inputs


print("Preprocessing Training Dataset:")
ds = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)

print("Preprocessing Evaluation Dataset:")
eval_ds = eval_ds.map(preprocess_function, batched=True, remove_columns=eval_ds.column_names)

data_collator = default_data_collator

train_dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

eval_dataloader = DataLoader(eval_ds, batch_size=BATCH_SIZE, collate_fn=data_collator)

def load_and_preprocess_data(tokenizer, split, limit, max_length):
    if split == "validation":
        dataset = load_dataset("code_search_net", "python", split=f"{split}[{limit*2}:{limit}]")
    else:
        dataset = load_dataset("code_search_net", "python", split=f"{split}[:{limit}]")

    def preprocess_function(examples):
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

    return dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)


# train_dataset = load_and_preprocess_data(tokenizer, "train", args.train_limit, args.max_length)
# valid_dataset = load_and_preprocess_data(tokenizer, "validation", args.valid_limit, args.max_length)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
t_total = len(train_dataloader) // GRAD_ACCUMULATION_STEPS * NUM_EPOCHS
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=adam_epsilon)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total)






global_step = 0
best_eval_loss = 0
patience = 5
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    student_model.train()
    total_loss = 0
    bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}")

    for step, batch in enumerate(bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        source_ids = batch["input_ids"]
        source_mask = batch["attention_mask"]
        target_ids = batch["labels"]

        # Teacher Predictions
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
            teacher_logits = teacher_outputs.logits

        # Student Predictions
        student_outputs = student_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
        student_logits = student_outputs.logits
        ce_loss = student_outputs.loss


         T = 2
         valid = (target_ids != -100).float()
         valid_tokens = valid.sum().clamp_min(1.0)

         s_logp = F.log_softmax(student_logits / T, dim=-1)
         t_prob = F.softmax(teacher_logits / T, dim=-1)

         kd = F.kl_div(s_logp, t_prob, reduction="none")
         kd = kd.sum(dim=-1)
         kd = (kd * valid).sum() / valid_tokens

         kd_loss = (T**2) * kd
         loss = 0.4 * ce_loss + 0.6 * kd_loss


         loss = loss / GRAD_ACCUMULATION_STEPS
         loss.backward()

         if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
             torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
             optimizer.step()
             scheduler.step()
             optimizer.zero_grad()
             global_step += 1

         bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")


     student_model.eval()
     eval_loss, tokens_num = 0, 0
     with torch.no_grad():
         for batch in tqdm(eval_dataloader, desc="Evaluating"):
             batch = {k: v.to(device) for k, v in batch.items()}
             outputs = student_model(**batch)
             eval_loss += outputs.loss.mean().item()

     eval_loss /= len(eval_dataloader)
     print(f"Epoch {epoch+1} Evaluation Loss: {eval_loss:.4f}")

     # Save Last Checkpoint
     os.makedirs(LAST_MODEL_DIR, exist_ok=True)
     torch.save(student_model.state_dict(), os.path.join(LAST_MODEL_DIR, "pytorch_model.bin"))

     # Save Best Checkpoint
     if eval_loss < best_eval_loss:
         best_eval_loss = eval_loss
         patience_counter = 0
         print("New Best Evaluation Loss Achieved!")
         os.makedirs(BEST_MODEL_DIR, exist_ok=True)
         torch.save(student_model.state_dict(), os.path.join(BEST_MODEL_DIR, "pytorch_model.bin"))
     else:
         patience_counter += 1
         print(f"No improvement for {patience_counter} epochs")

     # Early stopping check
     if patience_counter >= patience:
         print(f"Early stopping triggered after {patience} epochs without improvement")
         break

tokenizer.save_pretrained(OUTPUT_DIR)
print("Training Finished and Saved Successfully!")



best_model_path = os.path.join(BEST_MODEL_DIR, "pytorch_model.bin")
if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    student_model.load_state_dict(torch.load(best_model_path))
    model_for_inference = student_model
else:
    print("Warning: No best model found. Using current student model state.")
    print("This might result in poor performance if training didn't complete.")
    model_for_inference = student_model

model_for_inference.eval()


tokenizer.padding_side = "left"
prompts = []
references = []
test_imports_list = []
test_lists = []

print("\nExtracting Prompts from Validation Dataset:")
for sample in tqdm(es, desc="Fetching Prompts"):
    imports_and_signature = extract_function_signature(sample["code"])
    test_list_str = "\n".join(sample["test_list"])
    prompt = f"You are an expert Python programmer, and here is your task: {sample['prompt']}\nYour code should pass these tests and be syntactically correct:\n\n{test_list_str}\n[BEGIN]\n"
    if prompt:
        prompts.append(prompt)
        references.append(sample["code"])
        test_imports_list.append(sample["test_imports"])
        test_lists.append(sample["test_list"])

print("\nTokenizing Prompts for Validation:")
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    truncation=True,
    padding=True,
    return_attention_mask=True,
    max_length=512,
)
inputs = {k: v.to(device) for k, v in inputs.items()}

input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)


def generate_predictions(prompts, inputs, input_lengths, BATCH_SIZE, best_model, tokenizer):
    import psutil

    predictions = []

    process = psutil.Process(os.getpid())
    cpu_mem_before = process.memory_info().rss / (1024**2) 
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated() / (1024**2)
        gpu_mem_reserved_before = torch.cuda.memory_reserved() / (1024**2)
    else:
        gpu_mem_before = 0.0
        gpu_mem_reserved_before = 0.0

    print(f"[Memory] CPU memory before inference: {cpu_mem_before:.2f} MB")
    if torch.cuda.is_available():
        print(f"[Memory] GPU memory allocated before inference: {gpu_mem_before:.2f} MB")
        print(f"[Memory] GPU memory reserved before inference: {gpu_mem_reserved_before:.2f} MB")

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating Predictions"):
            batch_inputs = {
                "input_ids": inputs["input_ids"][i : i + BATCH_SIZE],
                "attention_mask": inputs["attention_mask"][i : i + BATCH_SIZE],
            }
            batch_input_lens = input_lengths[i : i + BATCH_SIZE]

            done_token_id = tokenizer.convert_tokens_to_ids("[DONE]")
            if done_token_id == tokenizer.unk_token_id:
                eos_token_for_generation = tokenizer.eos_token_id
                print("Warning: [DONE] token not found, using EOS token")
            else:
                eos_token_for_generation = done_token_id

            try:
                outputs = best_model.generate(
                    **batch_inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_for_generation,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            except Exception as e:
                print(f"Error during generation: {e}")
                outputs = best_model.generate(
                    **batch_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_for_generation,
                    return_dict_in_generate=True,
                )

            seqs = outputs.sequences
            for offset, seq in enumerate(seqs):
                full_text = tokenizer.decode(seq, skip_special_tokens=True)
                begin_marker = "[BEGIN]\n"
                begin_pos = full_text.rfind(begin_marker)
                if begin_pos != -1:
                    text = full_text[begin_pos + len(begin_marker) :]
                else:
                    start = batch_input_lens[offset].item()
                    text = tokenizer.decode(seq[start:], skip_special_tokens=True)

                print(f"Prompt length: {batch_input_lens[offset].item()}")
                print(f"Generated text length: {len(text)}")

                if len(text) > 50:
                    ascii_ratio = sum(
                        c.isascii() and (c.isalnum() or c in " \n\t()[]{}:,.\"'_-=+") for c in text
                    ) / len(text)
                    if ascii_ratio < 0.7:
                        print(f"Warning: Generated text might be gibberish (ASCII ratio: {ascii_ratio:.2f})")

                predictions.append(text)

    cpu_mem_after = process.memory_info().rss / (1024**2)  # in MB
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated() / (1024**2)
        gpu_mem_reserved_after = torch.cuda.memory_reserved() / (1024**2)
        gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        gpu_mem_after = 0.0
        gpu_mem_reserved_after = 0.0
        gpu_mem_peak = 0.0

    print(f"[Memory] CPU memory after inference: {cpu_mem_after:.2f} MB")
    print(f"[Memory] CPU memory used during inference: {cpu_mem_after - cpu_mem_before:.2f} MB")
    if torch.cuda.is_available():
        print(f"[Memory] GPU memory allocated after inference: {gpu_mem_after:.2f} MB")
        print(f"[Memory] GPU memory reserved after inference: {gpu_mem_reserved_after:.2f} MB")
        print(f"[Memory] GPU memory used during inference: {gpu_mem_after - gpu_mem_before:.2f} MB")
        print(f"[Memory] GPU peak memory allocated during inference: {gpu_mem_peak:.2f} MB")

    return predictions


tracker = EmissionsTracker()
tracker.start()


predictions = generate_predictions(prompts, inputs, input_lengths, BATCH_SIZE, model_for_inference, tokenizer)

emissions = tracker.stop()
print(f"\nCarbon emissions for inference: {emissions:.6f} kg CO2")

ref = references
pred = predictions
if len(ref) != len(pred):
    n = min(len(ref), len(pred))
    ref, pred = ref[:n], pred[:n]


def check_correctness(generated_code, test_imports, test_list):
    try:
        full_code = "\n".join(test_imports) + "\n" + generated_code
        with open("temp_code.py", "w") as f:
            f.write(full_code)
            for test in test_list:
                f.write("\n" + test)
        result = os.system("python temp_code.py")
        return result == 0
    except Exception:
        return False
    finally:
        if os.path.exists("temp_code.py"):
            os.remove("temp_code.py")


 pass_count = 0
 for i in tqdm(range(len(pred)), desc="Evaluating Functional Correctness"):
     if check_correctness(pred[i], test_imports_list[i], test_lists[i]):
         pass_count += 1

 pass_rate = (pass_count / len(pred)) * 100
 print(f"\nFunctional Correctness (Pass Rate): {pass_rate:.2f}%")

 print("\nComputing BLEU Score:")
 results = bleu.compute(predictions=pred, references=ref)
 print("BLEU-4 score:", results["bleu"])

 print("\nComputing CodeBLEU Score:")
 codebleu_result = calc_codebleu(
     references=ref,
     predictions=pred,
     lang="python",
     weights=(0.25, 0.25, 0.25, 0.25),
 )
 print("CodeBLEU score:", codebleu_result["codebleu"])

 print("\nComputing METEOR Score:")
 meteor_res = meteor.compute(predictions=pred, references=ref)
 print("METEOR:", meteor_res["meteor"])


 print("\nComputing ROUGE Scores:")
 rouge_res = rouge.compute(predictions=pred, references=ref, use_stemmer=True)
 print("ROUGE-1:", rouge_res.get("rouge1"))
 print("ROUGE-2:", rouge_res.get("rouge2"))
 print("ROUGE-L:", rouge_res.get("rougeL"))
 print("ROUGE-Lsum:", rouge_res.get("rougeLsum"))

 print("\nSaving Results to CSV File...")


 results_df = pd.DataFrame({"Prompt": prompts, "Test cases": test_lists, "Reference": ref, "Prediction": pred})

 csv_file_path = "distilled_qwen_2_5_coder_1_b_instruct_mbpp_results.csv"
 results_df.to_csv(csv_file_path, index=False, encoding="utf-8")
