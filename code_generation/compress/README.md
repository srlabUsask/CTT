### Fine-tune

To fine-tune decoder on the dataset. Change epoch for how long you want to run. For quicker result, it is now given as 1.

```
python run.py \
    --do_train \
    --output_dir codegen-350m-mono-student \
    --teacher_model_id Salesforce/codegen-350M-mono \
    --teacher_model_path ../finetune/model \
    --train_data_limit 70000 \
    --valid_data_limit 5000 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_length 512 \
    --n_head 4 \
    --n_layer 6 \
    --n_inner 128 \
    --n_embd 128 \
    --seed 42
```


### Inference

```
python run.py \
    --do_eval \
    --output_dir codegen-350m-mono-student \
    --teacher_model_id Salesforce/codegen-350M-mono \
    --test_data_limit 5000 \
    --batch_size 32 \
    --max_length 512 \
    --temp 0.8
```
