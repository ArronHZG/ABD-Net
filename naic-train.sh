python train.py -s naic \
    -t naic \
    --flip-eval --eval-freq 1 \
    --label-smooth \
    --criterion htri \
    --lambda-htri 0.1  \
    --data-augment crop random-erase \
    --margin 1.2 \
    --train-batch-size 48 \
    --height 384 \
    --width 128 \
    --optim adam --lr 0.0003 \
    --stepsize 20 40 \
    --gpu-devices 0,1 \
    --max-epoch 80 \
    --save-dir /home/arron/PycharmProjects/ABD-Net/model/naic \
    --arch resnet50 \
    --use-of \
    --abd-dan cam pam \
    --abd-np 2 \
    --shallow-cam \
    --use-ow \
    --workers 10 \
    --root /home/arron/dataset \
    --fixbase-epoch 10 
    # --evaluate




# ==> BatchTime(s)/BatchSize(img): 0.411/100
# Computing CMC and mAP
# Results ----------
# mAP: 66.55%
# CMC curve
# Rank-1  : 67.41%
# Rank-5  : 78.15%
# Rank-10 : 81.79%
# Rank-20 : 85.90%
# ------------------
