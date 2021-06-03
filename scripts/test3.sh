
model='bert-base-uncased' #110M, --gru_only
# model='bert-large-uncased' #336M, --gru_only
# model='albert-base-v2' #11M
# model='albert-large-v2' #17M
# model='albert-xlarge-v2' # 58M
# model='distilbert-base-uncased'


export CUDA_VISIBLE_DEVICES=0
bs=256
model='bert-base-uncased'
# python train2.py --gru_ep 20 --bert_ep 5 --gru_lr 5e-3 --bert_lr 1e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 20 --bert_ep 5 --gru_lr 1e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 20 --bert_ep 5 --gru_lr 5e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 20 --bert_ep 5 --gru_lr 1e-3 --bert_lr 1e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'

# python train2.py --gru_ep 5 --bert_ep 20 --gru_lr 5e-3 --bert_lr 1e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 5 --bert_ep 20 --gru_lr 1e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 5 --bert_ep 20 --gru_lr 5e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 5 --bert_ep 20 --gru_lr 1e-3 --bert_lr 1e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'

# export CUDA_VISIBLE_DEVICES=2
# python train.py --gru_ep 20 --gru_lr 3e-3 --bert_lr 3e-5 --bs $bs --pretrained_model $model

# export CUDA_VISIBLE_DEVICES=3
# python train.py --gru_ep 200 --gru_lr 3e-3 --bert_lr 3e-5 --bs $bs --pretrained_model $model


# python train2.py --gru_ep 0 --bert_ep 20 --gru_lr 3e-3 --bert_lr 1e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 0 --bert_ep 20 --gru_lr 3e-3 --bert_lr 3e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 0 --bert_ep 20 --gru_lr 3e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
# python train2.py --gru_ep 0 --bert_ep 20 --gru_lr 3e-3 --bert_lr 7e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'

python train2.py --gru_ep 5 --bert_ep 10 --gru_lr 3e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'