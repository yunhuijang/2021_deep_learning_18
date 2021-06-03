
model='bert-base-uncased' #110M, --gru_only
# model='bert-large-uncased' #336M, --gru_only
# model='albert-base-v2' #11M
# model='albert-large-v2' #17M
# model='albert-xlarge-v2' # 58M
# model='distilbert-base-uncased'

bs=256
# export CUDA_VISIBLE_DEVICES=0
# python train.py --gru_ep 20 --bert_lr 5e-5 --gru_lr 3e-3 --bs 256  --pretrained_model $model &
# export CUDA_VISIBLE_DEVICES=1
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 3e-5 --gru_lr 1e-3 --bs 256  --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=2
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 5e-5 --gru_lr 1e-3 --bs 256  --pretrained_model distilbert-base-uncased &
# wait

# export CUDA_VISIBLE_DEVICES=0
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 1e-5 --gru_lr 1e-2 --bs 256  --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=1
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 3e-5 --gru_lr 1e-2 --bs 256  --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=2
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 5e-5 --gru_lr 1e-2 --bs 256  --pretrained_model distilbert-base-uncased &
# wait

# export CUDA_VISIBLE_DEVICES=0
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 1e-5 --gru_lr 1e-3 --bs 256  --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=1
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 3e-5 --gru_lr 3e-3 --bs 256  --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=2
# python train.py --bert_ep 10 --gru_ep 20 --bert_lr 5e-5 --gru_lr 5e-3 --bs 256  --pretrained_model distilbert-base-uncased &
# wait

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

export CUDA_VISIBLE_DEVICES=0
bs=256

model='bert-base-uncased'
python train.py --gru_ep 10 --gru_lr 3e-3 --bert_lr 5e-5 --bs $bs --gru_only --pretrained_model $model --submitdir './submissions2'
# python train.py --gru_ep 10 --gru_lr 1e-3 --bs $bs --pretrained_model $model --gru_only --submitdir './submissions2'
# python train.py --gru_ep 10 --gru_lr 1e-2 --bs $bs --pretrained_model $model --gru_only --submitdir './submissions2'

# python train.py --gru_ep 100 --gru_lr 1e-4 --bs $bs --pretrained_model $model --gru_only --submitdir './submissions2'
# python train.py --gru_ep 100 --gru_lr 1e-3 --bs $bs --pretrained_model $model --gru_only --submitdir './submissions2'
# python train.py --gru_ep 100 --gru_lr 1e-2 --bs $bs --pretrained_model $model --gru_only --submitdir './submissions2'



# export CUDA_VISIBLE_DEVICES=2
# python train.py --gru_ep 20 --gru_lr 3e-3 --bert_lr 3e-5 --bs $bs --pretrained_model $model

# export CUDA_VISIBLE_DEVICES=3
# python train.py --gru_ep 200 --gru_lr 3e-3 --bert_lr 3e-5 --bs $bs --pretrained_model $model