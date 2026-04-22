import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import re
import evaluate
import torch
from codebleu import calc_codebleu
from tqdm import tqdm
import pandas as pd
import os

from codecarbon import EmissionsTracker


bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

BATCH_SIZE = 2
output_dir = "qwen2_5-coder-1_5b-tuned-code-mbpp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
eval_ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
es = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

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


few_shot_examples = ds.filter(lambda x: x["task_id"] in [2, 3, 4])
if len(few_shot_examples) < 3:
    print(f"Warning: Found only {len(few_shot_examples)} examples for few-shot prompt, expected 3.")


def extract_function_signature_for_prompt(code):
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


few_shot_prompt = ""
for example in few_shot_examples:
    prompt = example["prompt"]
    tests = "\n".join(example["test_list"])
    code = example["code"]
    fs = extract_function_signature_for_prompt(code)
    few_shot_prompt += f"You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n###IMPORTS AND FUNCTION SIGNATURE:\n{fs}\n[BEGIN]\n{code}\n[DONE]\n\n"


model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
teacher_model_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["[BEGIN]", "[DONE]"]})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    teacher_model_path, use_safetensors=True, trust_remote_code=True, device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))


def extract_function_signature(code):
    """
    Extract imports and function signature (def line) from the code.
    """
    import re

    lines = code.strip().split("\n")
    imports_and_signature = []

    # Extract all import statements
    for line in lines:
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            imports_and_signature.append(line)

    # Extract function signature
    for line in lines:
        line = line.strip()
        if line.startswith("def "):
            # Find the complete function signature (might span multiple lines)
            if line.endswith(":"):
                imports_and_signature.append(line)
                break
            else:
                # Handle multi-line function signatures
                func_sig = line
                for next_line in lines[lines.index(line) + 1 :]:
                    func_sig += " " + next_line.strip()
                    if next_line.strip().endswith(":"):
                        imports_and_signature.append(func_sig)
                        break
                break

    return "\n".join(imports_and_signature)


def preprocess_function(examples):
    function_signatures = [extract_function_signature(code) for code in examples["code"]]
    tests = ["\n".join(t) for t in examples["test_list"]]
    input_texts = [
        f"{few_shot_prompt}You are an expert Python programmer, and here is your task: {p} Your code should pass these tests:\n\n{t}\n###IMPORTS AND FUNCTION SIGNATURE:\n{fs}\n[BEGIN]\n"
        for p, t, fs in zip(examples["prompt"], tests, function_signatures)
    ]
    full_texts = [f"{inp}{code}\n[DONE]" for inp, code in zip(input_texts, examples["code"])
    model_inputs = tokenizer(full_texts, max_length=512, truncation=True, padding="max_length")
    labels = model_inputs["input_ids"].copy()
    for i in range(len(labels)):
        input_part = tokenizer(input_texts[i]).input_ids
        input_len = len(input_part)
        mask = [1] * len(labels[i])
        for j in range(input_len):
            if j < len(mask):
                mask[j] = 0

        labels[i] = [l if m == 1 else -100 for l, m in zip(labels[i], mask)]
        try:
            done_token_id = tokenizer.convert_tokens_to_ids("[DONE]")
            done_indices = [k for k, l in enumerate(labels[i]) if l == done_token_id]
            for k in done_indices:
                labels[i][k] = -100
        except:
            pass

    model_inputs["labels"] = labels
    return model_inputs



print("Preprocessing Training Dataset:")
ds = ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)

print("Preprocessing Evaluation Dataset:")
eval_ds = eval_ds.map(preprocess_function, batched=True, remove_columns=eval_ds.column_names)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    remove_unused_columns=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=2e-4,
    fp16=True,
    dataloader_pin_memory=False
    seed=42,
    disable_tqdm=False,
    ddp_find_unused_parameters=False,
    save_total_limit=1,
)

def train_model(model, tokenizer, train_dataset, valid_dataset, args):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        remove_unused_columns=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_strategy="epoch",
        disable_tqdm=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer


# train_dataset = load_and_preprocess_data(tokenizer, "train", args.train_limit, args.max_length)
# valid_dataset = load_and_preprocess_data(tokenizer, "validation", args.valid_limit, args.max_length)
# trainer = train_model(model, tokenizer, train_dataset, valid_dataset, args)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


print("\nTraining the Model:")
trainer.train()

print("\nLoading the Best Model:")
best_model = trainer.model


tokenizer.padding_side = "left"
prompts = []
references = []
test_imports_list = []
test_lists = []

print("\nExtracting Prompts from Validation Dataset:")
for sample in tqdm(es, desc="Fetching Prompts"):
    imports_and_signature = extract_function_signature(sample["code"])
    tests = "\n".join(sample["test_list"])
    prompt = f"{few_shot_prompt}You are an expert Python programmer, and here is your task: {sample['prompt']} Your code should pass these tests:\n\n{tests}\n###IMPORTS AND FUNCTION SIGNATURE:\n{imports_and_signature}\n[BEGIN]\n"
    if prompt:
        prompts.append(prompt)
        references.append(sample["code"])
        test_imports_list.append(sample["test_imports"])
        test_lists.append(sample["test_list"])
    if len(test_imports_list) > 150 and len(test_lists) > 150:
        break

print("\nTokenizing Prompts for Validation:")
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    truncation=True,
    padding=True,
    return_attention_mask=True,
    max_length=256,
)
inputs = {k: v.to(device) for k, v in inputs.items()}


input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)


def generate_predictions(prompts, inputs, input_lengths, BATCH_SIZE, best_model, tokenizer):
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating Predictions"):
            batch_inputs = {
                "input_ids": inputs["input_ids"][i : i + BATCH_SIZE],
                "attention_mask": inputs["attention_mask"][i : i + BATCH_SIZE],
            }
            batch_input_lens = input_lengths[i : i + BATCH_SIZE]

            outputs = best_model.generate(
                **batch_inputs,
                max_new_tokens=128,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("[DONE]"),
                return_dict_in_generate=True,
            )

            seqs = outputs.sequences
            for offset, seq in enumerate(seqs):
                start = batch_input_lens[offset].item()
                gen_tokens = seq[start:]
                text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                predictions.append(text)
    return predictions


tracker = EmissionsTracker()
tracker.start()

predictions = generate_predictions(prompts, inputs, input_lengths, BATCH_SIZE, best_model, tokenizer)

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
print("pass_count: ", pass_count)
pass_rate = (pass_count / len(pred)) * 100
print(f"\nFunctional Correctness (Pass Rate): {pass_rate:.2f}%")


print("\nComputing BLEU Score:")
bleu_refs = [[r] for r in ref]
bleu_res = bleu.compute(predictions=pred, references=bleu_refs)
print("BLEU-4 score:", bleu_res["bleu"])


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

results_df = pd.DataFrame({"Prompt": prompts, "Reference": ref, "Prediction": pred})

csv_file_path = "qwen_2_5-coder_1_5b_results_mbpp_new_prompt.csv"
results_df.to_csv(csv_file_path, index=False, encoding="utf-8")
