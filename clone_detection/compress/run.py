import os
import torch
import logging
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from models import Model, distill_loss
from utils import DistilledDataset
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel, RobertaTokenizer
from quanto import qint8, quantize
import time
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train(args, model, train_dataloader, eval_dataloader):
    num_steps = len(train_dataloader) * args.epochs
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps * 0.1, num_training_steps=num_steps
    )
    dev_best_acc = 0

    for epoch in range(args.epochs):
        model.train()
        tr_num = 0
        train_loss = 0

        logger.info("Epoch [{}/{}]".format(epoch + 1, args.epochs))
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        bar.set_description("Train")
        for batch in bar:
            texts = batch[0].to("cuda")
            soft_knowledge = batch[3].to("cuda")

            preds = model(texts)
            loss = distill_loss(preds, soft_knowledge)
            loss.backward()

            train_loss += loss.item()
            tr_num += 1

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_results = evaluate(model, eval_dataloader)
        dev_acc = dev_results["eval_acc"]
        if dev_acc >= dev_best_acc:
            dev_best_acc = dev_acc
            output_dir = os.path.join(args.model_dir, "best")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))
            logger.info("New best model found and saved.")
        else:
            output_dir = os.path.join(args.model_dir, "recent")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))

        logger.info(
            "Train Loss: {0}, Val Acc: {1}, Val Precision: {2}, Val Recall: {3}, Val F1: {4}".format(
                train_loss / tr_num,
                dev_results["eval_acc"],
                dev_results["eval_precision"],
                dev_results["eval_recall"],
                dev_results["eval_f1"],
            )
        )


def evaluate(model, eval_dataloader):
    eval_time_start = time.time()
    model.eval()
    predict_all = []
    labels_all = []

    with torch.no_grad():
        bar = tqdm(eval_dataloader, total=len(eval_dataloader))
        bar.set_description("Evaluation")
        for batch in bar:
            texts = batch[0].to("cuda")
            label = batch[1].to("cuda")

            cos_sim = model(texts)
            predict_all.append(cos_sim.cpu().numpy())
            labels_all.append(label.cpu().numpy())

    eval_time_end = time.time()
    print(f"Time for evaluation {eval_time_end-eval_time_start} seconds")
    predict_all = np.concatenate(predict_all, 0)
    labels_all = np.concatenate(labels_all, 0)

    preds = predict_all > 0.5
    recall = recall_score(labels_all, preds)
    precision = precision_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)

    results = {
        "eval_acc": np.mean(labels_all == preds),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
    }
    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--model_dir",
        default="./",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--choice", default="best", type=str, help="Model to test")
    parser.add_argument("--vocab_size", default=10000, type=int, help="Vocabulary Size.")
    parser.add_argument("--attention_heads", default=4, type=int, help="attention_heads")
    parser.add_argument("--hidden_dim", default=8, type=int, help="Hidden dim of student model.")
    parser.add_argument("--n_layers", default=8, type=int, help="Num of layers in student model.")
    parser.add_argument("--intermediate_size", default=8, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", type=int, default=42, help="number of epoch for training")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")

    args = parser.parse_args()

    start = time.time()

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    n_labels = 2
    config = RobertaConfig.from_pretrained("microsoft/unixcoder-base")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    config.num_labels = n_labels
    config.num_attention_heads = args.attention_heads
    config.hidden_size = args.hidden_dim
    config.intermediate_size = args.intermediate_size
    config.num_hidden_layers = args.n_layers
    config.hidden_dropout_prob = 0.2
    model = Model(RobertaModel(config=config), config, tokenizer, args)
    quantize(model, weights=qint8)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"{total_params:,} total parameters.")
    logger.info(f"{total_params*4/1e6} MB model size")

    if args.do_train:
        train_dataset = DistilledDataset(args, args.vocab_size, args.train_data_file, logger)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_dataset = DistilledDataset(args, args.vocab_size, args.eval_data_file, logger)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=8, pin_memory=True
    )

    model.to(args.device)

    if args.do_train:
        train(args, model, train_dataloader, eval_dataloader)

    if args.do_eval:
        model_dir = "../checkpoint/best/model.bin"
        model.load_state_dict(torch.load(model_dir), strict=False)
        model.to(args.device)
        tracker = EmissionsTracker()
	tracker.start()
        eval_res = evaluate(model, eval_dataloader)
        emissions = tracker.stop()
	print(f"\nCarbon emissions for inference: {emissions:.6f} kg CO2")
        logger.info(
            "Acc: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(
                eval_res["eval_acc"], eval_res["eval_precision"], eval_res["eval_recall"], eval_res["eval_f1"]
            )
        )
    end = time.time()
    print(f"total time: {end-start} seconds")


if __name__ == "__main__":
    main()
