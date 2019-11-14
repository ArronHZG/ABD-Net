python train.py -s naic \
    -t naic \
    --flip-eval --eval-freq 1 \
    --label-smooth \
    --criterion htri \
    --lambda-htri 0.1  \
    --margin 0.3 \
    --train-batch-size 32 \
    --height 384 \
    --width 128 \
    --optim adam --lr 0.0004 \
    --stepsize 20 40 \
    --gpu-devices 0,1 \
    --max-epoch 120 \
    --save-dir /home/arron/PycharmProjects/ABD-Net/model/naic/2 \
    --arch resnet50 \
    --use-of \
    --abd-dan cam pam \
    --abd-np 2 \
    --shallow-cam \
    --use-ow \
    --workers 10 \
    --root /home/arron/dataset \
    --fixbase-epoch 20 \
    --load-weights '/home/arron/PycharmProjects/ABD-Net/model/naic/2/checkpoint_ep23.pth.tar'


# ==> Test
# Evaluating naic ...
# # Using Flip Eval
# Extracted features for query set, obtained 1348-by-3072 matrix
# Extracted features for gallery set, obtained 2255-by-3072 matrix
# ==> BatchTime(s)/BatchSize(img): 0.183/100
# Computing CMC and mAP
# Results ----------
# mAP: 75.06%
# CMC curve
# Rank-1  : 75.63%
# Rank-5  : 84.69%
# Rank-10 : 88.42%
# Rank-20 : 91.13%
# ------------------
