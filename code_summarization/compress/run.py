import torch
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import os
import json
from quanto import qint8, quantize
import numpy as np
import bleu
from torch.utils.data.distributed import DistributedSampler
import random
import copy
import argparse
import logging
from codecarbon import EmissionsTracker

MODEL_CLASSES = {"codet5": (RobertaTokenizer, T5ForConditionalGeneration, T5Config)}
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


class Example(object):
    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            js = json.loads(line.strip())
            code = " ".join(js["code_tokens"]).replace("\n", " ")
            nl = " ".join(js["docstring_tokens"]).replace("\n", "")
            examples.append(Example(idx=idx, source=code, target=nl))
    return examples


class InputFeatures(object):
    def __init__(self, example_id, source_ids, target_ids, source_mask, target_mask):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, max_source_length, max_target_length, stage=None):
    features = []
    for example in examples:
        # source
        source_tokens = tokenizer.tokenize(example.source)[: max_source_length - 2]
        source_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + source_tokens + [tokenizer.sep_token])
        source_mask = [1] * len(source_ids)
        padding_length = max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        target_tokens = tokenizer.tokenize(example.target if stage != "test" else "None")[: max_target_length - 2]
        target_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + target_tokens + [tokenizer.sep_token])
        target_mask = [1] * len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        features.append(InputFeatures(example.idx, source_ids, target_ids, source_mask, target_mask))
    return features


def seq2seq_distill_loss(student_logits, teacher_logits, ce_loss, temperature=1, alpha=0.15):
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature**2)
    return alpha * kd_loss + (1 - alpha) * ce_loss


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate small model.")
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-base-multi-sum")
    parser.add_argument(
        "--model_type",
        type=str,
        default="codet5",
        choices=MODEL_CLASSES.keys(),
        help="Type of model architecture to use",
    )
    parser.add_argument("--teacher_model_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=512)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--do_train", action="store_true", help="Run fine-tuning / training")
    mode_group.add_argument("--do_eval", action="store_true", help="Run inference / evaluation only")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = copy.deepcopy(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    tokenizer_class, model_class, config_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    if args.do_train:
        teacher_model = model_class.from_pretrained(args.model_name)
        teacher_model.to(device)

    config = config_class.from_pretrained(args.model_name)
    config.d_model = args.d_model
    config.num_layers = args.num_layers
    config.num_heads = args.num_heads
    config.d_ff = args.d_ff

    student_model = model_class.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)

    quantize(student_model, weights=qint8)

    if args.do_eval:
        student_model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best-bleu/pytorch_model.bin"), strict=False)

    student_model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        student_model = DDP(student_model)
    elif n_gpu > 1:
        # multi-gpu training
        student_model = torch.nn.DataParallel(student_model)

    if args.do_train:
        # Prepare data
        train_examples = read_examples(args.train_data_path)
        train_features = convert_examples_to_features(
            train_examples,
            tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        train_data = TensorDataset(
            torch.tensor([f.source_ids for f in train_features], dtype=torch.long),
            torch.tensor([f.source_mask for f in train_features], dtype=torch.long),
            torch.tensor([f.target_ids for f in train_features], dtype=torch.long),
            torch.tensor([f.target_mask for f in train_features], dtype=torch.long),
        )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.batch_size // args.gradient_accumulation_steps
        )

        dev_dataset = {}
        eval_examples = read_examples(args.valid_data_path)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, max_source_length=args.max_source_length, max_target_length=args.max_target_length
        )
        eval_data = TensorDataset(
            torch.tensor([f.source_ids for f in eval_features], dtype=torch.long),
            torch.tensor([f.source_mask for f in eval_features], dtype=torch.long),
            torch.tensor([f.target_ids for f in eval_features], dtype=torch.long),
            torch.tensor([f.target_mask for f in eval_features], dtype=torch.long),
        )
        dev_dataset["dev_loss"] = eval_examples, eval_data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    if args.do_eval:
        test_dataset = {}
        test_examples = read_examples(args.test_data_path)
        test_features = convert_examples_to_features(
            test_examples, tokenizer, max_source_length=args.max_source_length, max_target_length=args.max_target_length
        )
        test_data = TensorDataset(
            torch.tensor([f.source_ids for f in test_features], dtype=torch.long),
            torch.tensor([f.source_mask for f in test_features], dtype=torch.long),
            torch.tensor([f.target_ids for f in test_features], dtype=torch.long),
            torch.tensor([f.target_mask for f in test_features], dtype=torch.long),
        )
        test_dataset["test_loss"] = test_examples, test_data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=256)

    if args.do_train:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epoch
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

        student_model.train()
        teacher_model.eval()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num epoch = %d", args.num_train_epoch)

        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, float("inf")
        total_loss = 0
        for epoch in range(args.num_train_epoch):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to("cuda") for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch

                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
                    teacher_logits = teacher_outputs.logits

                # Student predictions
                student_outputs = student_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
                student_logits = student_outputs.logits
                ce_loss = student_outputs.loss

                # Knowledge Distillation Loss
                loss = seq2seq_distill_loss(student_logits, teacher_logits, ce_loss)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            # Eval model with dev dataset
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            eval_flag = False

            print("eval started")

            # Start Evaling model
            student_model.eval()
            eval_loss, tokens_num = 0, 0
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch

                with torch.no_grad():
                    student_outputs = student_model(input_ids=source_ids, attention_mask=source_mask, labels=target_ids)
                    active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
                    eval_loss += student_outputs.loss.sum().item() * active_loss.sum()
                    tokens_num += active_loss.sum()

            # Pring loss of dev dataset
            student_model.train()
            eval_loss = eval_loss / tokens_num
            eval_loss = eval_loss.cpu().item()
            result = {
                "eval_ppl": round(np.exp(eval_loss), 5),
                "global_step": global_step + 1,
                "train_loss": round(train_loss, 5),
            }

            for key in sorted(result.keys()):
                print(key, " = ", str(result[key]))

            last_output_dir = os.path.join(output_dir, "checkpoint-last")
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = (
                student_model.module if hasattr(student_model, "module") else student_model
            )  # Only save the model it-self
            output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            if eval_loss < best_loss:
                print("  Best ppl: ", str(round(np.exp(eval_loss), 5)))
                best_loss = eval_loss
                output_dir = os.path.join(output_dir, "checkpoint-best-ppl")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    student_model.module if hasattr(student_model, "module") else student_model
                )
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            output_dir = copy.deepcopy(args.output_dir)


            print("bleu score started")
            eval_examples, eval_data = dev_dataset["dev_loss"]

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)

            student_model.eval()
            predictions = []
            p = []

            # Generate predictions for each batch
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                with torch.no_grad():
                    output_ids = student_model.module.generate(
                        input_ids=source_ids, attention_mask=source_mask, max_length=128
                    )

                # Decode predictions and store
                for i in range(output_ids.size(0)):
                    pred_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                    p.append(pred_text)

            # Save predictions and references for BLEU calculation
            with open(os.path.join(output_dir, "dev.output"), "w") as f, open(
                os.path.join(output_dir, "dev.gold"), "w"
            ) as f1:
                for ref, gold in zip(p, eval_examples):
                    predictions.append(str(gold.idx) + "\t" + ref)
                    f.write(str(gold.idx) + "\t" + ref + "\n")
                    f1.write(str(gold.idx) + "\t" + gold.target + "\n")

            (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(output_dir, "dev.gold"))
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            print("BLEU:", dev_bleu)
            student_model.train()

            if dev_bleu > best_bleu:
                print("Best bleu: ", str(dev_bleu))
                best_bleu = dev_bleu
                output_dir = os.path.join(output_dir, "checkpoint-best-bleu")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    student_model.module if hasattr(student_model, "module") else student_model
                )
                output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
            output_dir = copy.deepcopy(args.output_dir)

    if args.do_eval:
        student_model.eval()
        predictions = []
        p = []
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            with torch.no_grad():
                output_ids = student_model.module.generate(input_ids=source_ids, attention_mask=source_mask)

            for i in range(output_ids.size(0)):
                pred_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                p.append(pred_text)

        with open(os.path.join(output_dir, "dev_test.output"), "w") as f, open(
            os.path.join(output_dir, "dev_test.gold"), "w"
        ) as f1:
            for ref, gold in zip(p, test_examples):
                predictions.append(str(gold.idx) + "\t" + ref)
                f.write(str(gold.idx) + "\t" + ref + "\n")
                f1.write(str(gold.idx) + "\t" + gold.target + "\n")

        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(output_dir, "dev_test.gold"))
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        print("BLEU:", dev_bleu)


if __name__ == "__main__":
    main()
