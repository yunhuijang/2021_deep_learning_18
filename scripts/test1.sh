


export CUDA_VISIBLE_DEVICES=0
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 1e-5 --gru_lr 1e-3 --bs 256  --pretrained_model distilbert-base-uncased &
export CUDA_VISIBLE_DEVICES=1
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 3e-5 --gru_lr 1e-3 --bs 256  --pretrained_model distilbert-base-uncased &
export CUDA_VISIBLE_DEVICES=2
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 5e-5 --gru_lr 1e-3 --bs 256  --pretrained_model distilbert-base-uncased &
wait

export CUDA_VISIBLE_DEVICES=0
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 1e-5 --gru_lr 1e-2 --bs 256  --pretrained_model distilbert-base-uncased &
export CUDA_VISIBLE_DEVICES=1
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 3e-5 --gru_lr 1e-2 --bs 256  --pretrained_model distilbert-base-uncased &
export CUDA_VISIBLE_DEVICES=2
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 5e-5 --gru_lr 1e-2 --bs 256  --pretrained_model distilbert-base-uncased &
wait

export CUDA_VISIBLE_DEVICES=0
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 1e-5 --gru_lr 1e-3 --bs 256  --pretrained_model distilbert-base-uncased &
export CUDA_VISIBLE_DEVICES=1
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 3e-5 --gru_lr 3e-3 --bs 256  --pretrained_model distilbert-base-uncased &
export CUDA_VISIBLE_DEVICES=2
python train.py --bert_ep 10 --gru_ep 20 --bert_lr 5e-5 --gru_lr 5e-3 --bs 256  --pretrained_model distilbert-base-uncased &
wait

# export CUDA_VISIBLE_DEVICES=0
# python train.py --ep 20 --bs 256 --bert_lr 3e-5 --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=1
# python train.py --ep 20 --bs 256 --bert_lr 3e-5 --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=2
# python train.py --ep 20 --bs 256 --bert_lr 3e-5 --pretrained_model distilbert-base-uncased &


# export CUDA_VISIBLE_DEVICES=0
# python train.py --bs 256 --lr 3e-3 --pretrained_model bert-large-uncased &
# export CUDA_VISIBLE_DEVICES=1
# python train.py --bs 256 --lr 3e-3 --pretrained_model bert-base-cased &
# export CUDA_VISIBLE_DEVICES=2
# python train.py --bs 256 --lr 3e-3 --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=3
# python train.py --bs 256 --lr 3e-3 --pretrained_model distilbert-base-cased &
# wait
