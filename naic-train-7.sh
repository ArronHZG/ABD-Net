# Dataset statistics:
#   ----------------------------------------
#   subset   | # ids | # images | # cameras
#   ----------------------------------------
#   train    |  4768 |    20429 |         1
#   query    |  1348 |     1348 |         1
#   gallery  |  1071 |     2255 |         1
#   ----------------------------------------
# !!! Using RandomIdentitySampler !!!
# => Initializing TEST (target) datasets
# => Naic loaded
# Dataset statistics:
#   ----------------------------------------
#   subset   | # ids | # images | # cameras
#   ----------------------------------------
#   train    |  4768 |    20429 |         1
#   query    |  1348 |     1348 |         1
#   gallery  |  1071 |     2255 |         1
#   ----------------------------------------


#   **************** Summary ****************
#   train names      : ['naic']
#   # train datasets : 1
#   # train ids      : 4768
#   # train images   : 20429
#   # train cameras  : 1
#   test names       : ['naic']
#   *****************************************

python train.py \
    --root /home/arron/dataset -s naic -t naic  -j 16 \
    --data-augment crop color-jitter random-erase \
    --optim adam --lr 5e-4 \
    --start-epoch 0 --max-epoch 150 \
    --train-batch-size 36 \
    --test-batch-size 64 \
    --fixbase-epoch 10 \
    --label-smooth \
    --criterion htri \
    --margin 1.2 \
    --num-instances 4 \
    --lambda-htri 0.1 \
    --arch resnet50 \
    --flip-eval --eval-freq 1 \
    --gpu-devices 0 \
    --branches global abd \
    --global-max-pooling \
    --shallow-cam \
    --np-with-global \
    --np-max-pooling \
    --abd-np 2 \
    --abd-dan cam pam \
    --use-of \
    --of-start-epoch 30 \
    --use-ow \
    --save-dir /home/arron/PycharmProjects/ABD-Net/model/naic/5 \
#    --resume /home/arron/PycharmProjects/ABD-Net/model/naic/5/checkpoint_best.pth.tar
#    --evaluate
#    --visualize-ranks