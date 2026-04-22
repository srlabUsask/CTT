### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0

### Fine-tune

To fine-tune encoder-decoder on the dataset. Change epoch for how long you want to run. For quicker result, it is now given as 1.

```
python run.py \
    --do_train \
    --do_eval \
    --model_type codet5 \
    --model_name_or_path Salesforce/codet5-base-multi-sum \
    --train_filename ../dataset/java/train.jsonl \
    --dev_filename ../dataset/java/valid.jsonl \
    --output_dir model/java \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 1
```


### Inference

```
python run.py \
    --do_test \
    --model_type codet5 \
    --model_name_or_path Salesforce/codet5-base-multi-sum \
    --load_model_path model/java/checkpoint-best-bleu/pytorch_model.bin \
    --test_filename ../dataset/java/test.jsonl \
    --output_dir model/java \
    --max_source_length 256 \
    --max_target_length 128 \
    --beam_size 10 \
    --eval_batch_size 64
```
