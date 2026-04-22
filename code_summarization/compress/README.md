### Fine-tune

To fine-tune encoder-decoder on the dataset. Change epoch for how long you want to run. For quicker result, it is now given as 1.

```
python run.py \
    --do_train \
    --train_data_path ../dataset/java/train.jsonl \
    --valid_data_path ../dataset/java/valid.jsonl \
    --teacher_model_path ../finetune/model/java/checkpoint-best-bleu/pytorch_model.bin \
    --output_dir model/java \
    --model_name Salesforce/codet5-base-multi-sum \
    --model_type codet5 \
    --d_model 128 \
    --num_layers 8 \
    --num_heads 6 \
    --d_ff 512 \
    --num_train_epoch 1 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --max_source_length 256 \
    --max_target_length 128 \
    --seed 42 \
```


### Inference

```
python run.py \
    --do_eval \
    --test_data_path ../dataset/java/test.jsonl \
    --output_dir model/java
```
