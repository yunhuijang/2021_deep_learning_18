


export CUDA_VISIBLE_DEVICES=0
python train.py --bs 256 --lr 2e-3 --pretrained_model bert-base-uncased &
export CUDA_VISIBLE_DEVICES=1
python train.py --bs 256 --lr 2e-3 --pretrained_model bert-base-cased &
export CUDA_VISIBLE_DEVICES=2
python train.py --bs 256 --lr 2e-3 --pretrained_model distilbert-base-uncased &
export CUDA_VISIBLE_DEVICES=3
python train.py --bs 256 --lr 2e-3 --pretrained_model distilbert-base-cased &
wait


# export CUDA_VISIBLE_DEVICES=0
# python train.py --bs 256 --lr 3e-3 --pretrained_model bert-large-uncased &
# export CUDA_VISIBLE_DEVICES=1
# python train.py --bs 256 --lr 3e-3 --pretrained_model bert-base-cased &
# export CUDA_VISIBLE_DEVICES=2
# python train.py --bs 256 --lr 3e-3 --pretrained_model distilbert-base-uncased &
# export CUDA_VISIBLE_DEVICES=3
# python train.py --bs 256 --lr 3e-3 --pretrained_model distilbert-base-cased &
# wait
