
model='bert-base-uncased' #110M
# model='distilbert-base-uncased'


export CUDA_VISIBLE_DEVICES=1
bs=256


model='bert-base-uncased'
python train_plain_bert.py --bert_ep 10 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
python train.py --gru_ep 10 --gru_lr 3e-3 --bs $bs --gru_only --pretrained_model $model --submitdir './submissions2'
python train.py --gru_ep 10 --gru_lr 3e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
python train2.py --gru_ep 5 --bert_ep 10 --gru_lr 3e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'



model='distilbert-base-uncased'
python train_plain_bert.py --bert_ep 10 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
python train.py --gru_ep 10 --gru_lr 3e-3 --bs $bs --gru_only --pretrained_model $model --submitdir './submissions2'
python train.py --gru_ep 10 --gru_lr 3e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
python train2.py --gru_ep 5 --bert_ep 10 --gru_lr 3e-3 --bert_lr 5e-5 --bs $bs --pretrained_model $model --submitdir './submissions2'
