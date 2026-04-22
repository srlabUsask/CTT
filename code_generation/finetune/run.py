import os
import re
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import evaluate
from codebleu import calc_codebleu
from codecarbon import EmissionsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Code Generation")
    parser.add_argument("--model_id", type=str, default="Salesforce/codegen-350M-multi")
    parser.add_argument("--output_dir", type=str, default="model")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=0.8)

    parser.add_argument("--train_limit", type=int)
    parser.add_argument("--valid_limit", type=int)
    parser.add_argument("--test_limit", type=int)

    parser.add_argument("--do_train", action="store_true", help="Train the model")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation only")

    return parser.parse_args()


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


def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return model, tokenizer


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


def extract_prompt(code_text):
    """
    Fetches code from the function definition line
    up until the end of the triple-quoted docstring.
    """
    pattern = r"(def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:[\s\S]*?" r"(\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''))"
    match = re.search(pattern, code_text)
    if match:
        return match.group(0)
    return None


def inference(model, tokenizer, test_data, max_length, batch_size, temp):
    prompts, references = [], []

    for sample in tqdm(test_data, desc="Extracting prompts"):
        prompt = extract_prompt(sample["func_code_string"])
        if prompt:
            prompts.append(prompt)
            references.append(sample["func_code_string"])

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        return_attention_mask=True,
        max_length=max_length,
    ).to("cuda")

    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating predictions"):
            batch_inputs = {
                "input_ids": inputs.input_ids[i : i + batch_size],
                "attention_mask": inputs.attention_mask[i : i + batch_size],
            }
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_length,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
            batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(batch_results)

    return prompts, references, predictions


def evaluate_metrics(predictions, references):
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]

    codebleu_score = calc_codebleu(
        references=references,
        predictions=predictions,
        lang="python",
        weights=(0.25, 0.25, 0.25, 0.25),
    )["codebleu"]

    return bleu_score, codebleu_score


def save_results(prompts, references, predictions, file_path="codegen_results.csv"):
    df = pd.DataFrame(
        {
            "Prompt": prompts,
            "Reference": references,
            "Prediction": predictions,
        }
    )
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Results saved to {file_path}")


def main():
    args = parse_args()
    assert args.do_train or args.do_eval, "Specify --do_train or --do_eval"

    model, tokenizer = get_model_and_tokenizer(args.model_id)

    if args.do_train:
        print("\n=== Training Mode ===")
        train_dataset = load_and_preprocess_data(tokenizer, "train", args.train_limit, args.max_length)
        valid_dataset = load_and_preprocess_data(tokenizer, "validation", args.valid_limit, args.max_length)
        trainer = train_model(model, tokenizer, train_dataset, valid_dataset, args)

    if args.do_eval:
        print("\n=== Evaluation Mode ===")
        test_dataset = load_dataset("code_search_net", "python", split=f"validation[:{args.test_limit}]")
        model = AutoModelForCausalLM.from_pretrained(args.output_dir).to("cuda")
        tracker = EmissionsTracker()
	tracker.start()
        prompts, refs, preds = inference(model, tokenizer, test_dataset, args.max_length, args.batch_size, args.temp)
        emissions = tracker.stop()
	print(f"\nCarbon emissions for inference: {emissions:.6f} kg CO2")

        print("\n=== Metrics ===")
        bleu_score, codebleu_score = evaluate_metrics(preds, refs)
        print(f"BLEU-4 score: {bleu_score}")
        print(f"CodeBLEU score: {codebleu_score}")

        save_results(prompts, refs, preds)


if __name__ == "__main__":
    main()
