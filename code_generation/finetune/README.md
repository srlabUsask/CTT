### Fine-tune

To fine-tune decoder on the dataset. Change epoch for how long you want to run. For quicker result, it is now given as 1.

```
python run.py \
    --do_train \
    --model_id Salesforce/codegen-350M-mono \
    --output_dir model/codegen \
    --train_limit 50000 \
    --valid_limit 2500 \
    --batch_size 32 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --seed 42
```


### Inference

```
python run.py \
    --do_eval \
    --model_id Salesforce/codegen-350M-mono \
    --output_dir model/codegen \
    --test_limit 5000 \
    --batch_size 32 \
    --max_length 512 \
    --temp 0.8
```
