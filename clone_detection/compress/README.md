In our paper, the architecture-related hyperparamenters is `{'attention_heads': 4, 'hidden_dim': 8, 'intermediate_size': 8, 'n_layers': 8}`.

To train, please run following. Change the epoch based on how long you want to train. For quicker result, it is now given as 1:
```
python run.py \
    --do_train \
    --train_data_file=../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../data/clone_detection/valid_sampled.txt \
    --model_dir ../checkpoint \
    --attention_heads 4 \
    --hidden_dim 8 \
    --intermediate_size 8 \
    --n_layers 8 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 1
```


To eval:
```
python run.py \
    --do_eval \
    --train_data_file=../data/clone_detection/unlabel_train.txt \
    --eval_data_file=../data/clone_detection/test_sampled.txt \
    --model_dir ../checkpoint \
    --attention_heads 4 \
    --hidden_dim 8 \
    --intermediate_size 8 \
    --n_layers 8 \
    --vocab_size 1000 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --epochs 1
```