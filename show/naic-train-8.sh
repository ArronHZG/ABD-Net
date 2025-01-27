python summary.py \
    --root /home/arron/dataset -s naic -t naic  -j 16 \
    --data-augment crop color-jitter random-erase \
    --optim adam --lr 1e-3 \
    --start-epoch 0 --max-epoch 150 \
    --train-batch-size 60 \
    --test-batch-size 64 \
    --fixbase-epoch 20 \
    --label-smooth \
    --criterion htri \
    --margin 1.2 \
    --num-instances 4 \
    --lambda-htri 0.1 \
    --arch resnet50 \
    --flip-eval --eval-freq 1 \
    --gpu-devices 1 \
    --branches global abd \
    --global-max-pooling \
    --shallow-cam \
    --np-with-global \
    --np-max-pooling \
    --abd-np 2 \
    --abd-dan cam pam \
    --use-of \
    --of-start-epoch 50 \
    --use-ow \