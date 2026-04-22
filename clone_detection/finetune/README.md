## Evaluation
For evaluating the fine-tuned model, please run:
```
python main.py \
    --do_eval \
    --train_data_file=../data/clone_detection/label_train.txt \
    --eval_data_file=../data/clone_detection/test_sampled.txt \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 123456
```

## Finetuning
To finetune, please run. Change the epoch based on how long you want to train. For quicker result, it is now given as 1:
```
python main.py \
    --do_train \
    --train_data_file=../data/clone_detection/label_train.txt \
    --eval_data_file=../data/clone_detection/valid_sampled.txt \
    --epoch 1 \
    --block_size 512 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

## Get soft labels
For getting soft labels, please run:
```
python main.py \
    --do_eval \
    --train_data_file=../data/clone_detection/label_train.txt \
    --eval_data_file=../data/clone_detection/unlabel_train.txt \
    --block_size 400 \
    --eval_batch_size 64 \
    --seed 123456
```