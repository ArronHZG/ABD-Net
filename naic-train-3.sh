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


10 (fix 10)
# Results ----------
# mAP: 76.73%
# CMC curve
# Rank-1  : 76.84%
# Rank-5  : 87.11%
# Rank-10 : 89.54%
# Rank-20 : 93.28%
# ------------------
# Save! 0 0.7684407
# Finished. Total elapsed time (h:m:s): 0:23:42. Training time (h:m:s): 0:23:25.
# => Show summary
# naic (source)
# - epoch 10	 rank1 76.8%
# ==========


python train.py \
    --root /home/arron/dataset -s naic -t naic  -j 8 \
    --data-augment crop,color-jitter,random-erase \
    --optim adam --lr 5e-5 \
    --start-epoch 60 --max-epoch 100 \
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
    --flip-eval --eval-freq 1 \
    --gpu-devices 0,1 \
    --branches global, abd, dan, np \
    --global-max-pooling \
    --shallow-cam \
    --np-with-global \
    --np-max-pooling \
    --use-of \
    --of-start-epoch 100 \
    --use-ow \
    --save-dir /home/arron/PycharmProjects/ABD-Net/model/naic/3 \
    --load-weights /home/arron/PycharmProjects/ABD-Net/model/naic/3/checkpoint_ep60.pth.tar
#    --evaluate
#    --visualize-ranks

20
Results ----------
mAP: 77.38%
CMC curve
Rank-1  : 76.84%
Rank-5  : 86.83%
Rank-10 : 90.85%
Rank-20 : 93.00%
------------------
Save! 0 0.7684407

30
Results ----------
mAP: 79.85%
CMC curve
Rank-1  : 79.08%
Rank-5  : 89.17%
Rank-10 : 92.16%
Rank-20 : 94.40%
------------------
40
Results ----------
mAP: 82.01%
CMC curve
Rank-1  : 81.42%
Rank-5  : 90.85%
Rank-10 : 93.18%
Rank-20 : 95.42%
------------------
50
Results ----------
mAP: 81.82%
CMC curve
Rank-1  : 81.05%
Rank-5  : 90.48%
Rank-10 : 93.09%
Rank-20 : 95.52%
------------------
60
Results ----------
mAP: 82.34%
CMC curve
Rank-1  : 81.79%
Rank-5  : 90.66%
Rank-10 : 93.00%
Rank-20 : 95.42%
------------------
Save! 0.81419235 0.8179272
Finished. Total elapsed time (h:m:s): 1:21:01. Training time (h:m:s): 1:19:44.
=> Show summary
naic (source)
- epoch 20	 rank1 76.8%
- epoch 30	 rank1 79.1%
- epoch 40	 rank1 81.4%
- epoch 50	 rank1 81.0%
- epoch 60	 rank1 81.8%
==========