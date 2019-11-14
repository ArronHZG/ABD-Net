python train.py \
    --root /home/arron/dataset -s naic -t naic  -j 8 \
    --data-augment crop,color-jitter,random-erase \
    --optim adam --lr 0.0005 \
    --start-epoch 0 --max-epoch 10 \
    --stepsize 20 40 60 80 100 120 140 \
    --train-batch-size 160 \
    --test-batch-size 64 \
    --fixbase-epoch 10 \
    --label-smooth \
    --criterion htri \
    --margin 0.3 \
    --num-instances 4 \
    --lambda-htri 0.1  \
    --arch resnet50 \
    --flip-eval --eval-freq 10 \
    --gpu-devices 0,1 \
    --branches global, abd, dan, np \
    --global-max-pooling \
    --shallow-cam \
    --np-with-global \
    --np-max-pooling \
    --use-of \
    --of-start-epoch 60 \
    --use-ow \
    --save-dir /home/arron/PycharmProjects/ABD-Net/model/naic/3 \
#    --load-weights
#    --evaluate



python train.py \
    --root /home/arron/dataset -s naic -t naic  -j 8 \
    --data-augment crop,color-jitter,random-erase \
    --optim adam --lr 0.0005 \
    --start-epoch 10 --max-epoch 60 \
    --stepsize 20 40 60 80 100 120 140 \
    --train-batch-size 64 \
    --test-batch-size 64 \
    --fixbase-epoch 0 \
    --label-smooth \
    --criterion htri \
    --margin 0.3 \
    --num-instances 4 \
    --lambda-htri 0.1  \
    --arch resnet50 \
    --flip-eval --eval-freq 10 \
    --gpu-devices 0,1 \
    --branches global, abd, dan, np \
    --global-max-pooling \
    --shallow-cam \
    --np-with-global \
    --np-max-pooling \
    --use-of \
    --of-start-epoch 60 \
    --use-ow \
    --save-dir /home/arron/PycharmProjects/ABD-Net/model/naic/3 \
    --load-weights /home/arron/PycharmProjects/ABD-Net/model/naic/3/checkpoint_ep10.pth.tar
#    --evaluate



python train.py \
    --root /home/arron/dataset -s naic -t naic  -j 8 \
    --data-augment crop,color-jitter,random-erase \
    --optim adam --lr 0.0005 \
    --start-epoch 60 --max-epoch 150 \
    --stepsize 20 40 60 80 120 140 \
    --train-batch-size 32 \
    --test-batch-size 64 \
    --fixbase-epoch 0 \
    --label-smooth \
    --criterion htri \
    --margin 0.3 \
    --num-instances 4 \
    --lambda-htri 0.1  \
    --arch resnet50 \
    --flip-eval --eval-freq 10 \
    --gpu-devices 0,1 \
    --branches global, abd, dan, np \
    --global-max-pooling \
    --shallow-cam \
    --np-with-global \
    --np-max-pooling \
    --use-of \
    --of-start-epoch 10 \
    --use-ow \
    --save-dir /home/arron/PycharmProjects/ABD-Net/model/naic/3 \
    --load-weights /home/arron/PycharmProjects/ABD-Net/model/naic/3/checkpoint_ep60.pth.tar
#    --evaluate

