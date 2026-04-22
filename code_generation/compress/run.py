import os
import re
import argparse
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
import evaluate
from codebleu import calc_codebleu
from quanto import quantize, qfloat8
import copy
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--output_dir", type=str, default="codegen-350m-mono-student")
    parser.add_argument("--teacher_model_id", type=str, default="Salesforce/codegen-350M-mono")
    parser.add_argument("--teacher_model_path", type=str, default="../finetune/model")
    parser.add_argument("--train_data_limit", type=int, default=70000)
    parser.add_argument("--valid_data_limit", type=int, default=5000)
    parser.add_argument("--test_data_limit", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_inner", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_function(examples, tokenizer, max_length):
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


def fetch_code_until_docstring_ends(code_text):
    """
    Fetches code from the function definition line
    up until the end of the triple-quoted docstring.
    """
    pattern = r"(def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:[\s\S]*?" r"(\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''))"
    match = re.search(pattern, code_text)
    if match:
        return match.group(0)
    return None


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bleu = evaluate.load("bleu")

    BEST_MODEL_DIR = os.path.join(args.output_dir, "best_model")
    LAST_MODEL_DIR = os.path.join(args.output_dir, "last_model")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.do_train:
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path).to(device)
        teacher_model.to(device)
    
    base_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_id)
    student_config = copy.deepcopy(base_model.config)
    student_config.n_head = args.n_head
    student_config.n_layer = args.n_layer
    student_config.n_inner = args.n_inner
    student_config.n_embd = args.n_embd

    student_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_id, config=student_config, ignore_mismatched_sizes=True
    )
    quantize(student_model, weights=qfloat8)

    if args.do_eval:
        student_model.load_state_dict(torch.load(os.path.join(BEST_MODEL_DIR, "pytorch_model.bin")), strict=False)
        
    student_model.to(device)

    if torch.cuda.device_count() > 1:
        if args.do_train:
            teacher_model = DataParallel(teacher_model)
        student_model = DataParallel(student_model)

    if args.do_train:
        train_data = load_dataset("code_search_net", "python", split=f"train[:{args.train_data_limit}]")
        valid_data = load_dataset("code_search_net", "python", split=f"validation[{args.valid_data_limit}:{args.valid_data_limit*2}]")
        train_data = train_data.map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=train_data.column_names,
        )

        valid_data = valid_data.map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=valid_data.column_names,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
        eval_dataloader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=data_collator)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        t_total = len(train_dataloader) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total)

        global_step = 0
        best_eval_loss = float("inf")

        for epoch in range(args.num_train_epochs):
            print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
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
                    teacher_outputs = teacher_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
                    teacher_logits = teacher_outputs.logits

                # Student Predictions
                student_outputs = student_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
                student_logits = student_outputs.logits
                ce_loss = student_outputs.loss

                # Knowledge Distillation Loss
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / args.kd_temp, dim=-1), F.softmax(teacher_logits / args.kd_temp, dim=-1), reduction="batchmean"
                )
                loss = 0.6 * kd_loss + 0.4 * ce_loss
                if loss.dim() > 0:
                    loss = loss.mean()

                total_loss += loss.item()
                loss.backward()

                if (step + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

            student_model.eval()
            eval_loss = 0
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
                print("New Best Evaluation Loss Achieved!")
                os.makedirs(BEST_MODEL_DIR, exist_ok=True)
                torch.save(student_model.state_dict(), os.path.join(BEST_MODEL_DIR, "pytorch_model.bin"))


    if args.do_eval:
        test_data = load_dataset("code_search_net", "python", split=f"validation[:{args.test_data_limit}]")
        model_for_inference = student_model.module if hasattr(student_model, "module") else student_model
        model_for_inference.eval()

        prompts, references = [], []
        for sample in tqdm(test_data, desc="Preparing Prompts"):
            prompt = fetch_code_until_docstring_ends(sample["func_code_string"])
            if prompt:
                prompts.append(prompt)
                references.append(sample["func_code_string"])

        print("\nTokenizing Prompts for Validation:")
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_attention_mask=True,
            max_length=args.max_length,
        ).to(device)

        print("\nRunning Batch Inference:")
        tracker = EmissionsTracker()
	tracker.start()
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating Predictions"):
                batch_inputs = {
                    "input_ids": inputs.input_ids[i : i + args.batch_size],
                    "attention_mask": inputs.attention_mask[i : i + args.batch_size],
                }
                outputs = model_for_inference.generate(
                    **batch_inputs,
                    max_new_tokens=args.max_length,
                    temperature=args.temp,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                )
                batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(batch_results)
	emissions = tracker.stop()
	print(f"\nCarbon emissions for inference: {emissions:.6f} kg CO2")
        results = bleu.compute(predictions=predictions, references=references)
        print("BLEU-4 score:", results["bleu"])

        codebleu_result = calc_codebleu(
            references=references,
            predictions=predictions,
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25),
        )
        print("CodeBLEU score:", codebleu_result["codebleu"])

        results_df = pd.DataFrame({"Prompt": prompts, "Reference": references, "Prediction": predictions})
        results_df.to_csv("student_code_gen_results.csv", index=False, encoding="utf-8")
        print("Results saved to student_code_gen_results.csv")


if __name__ == "__main__":
    main()
