# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine
from torch.backends import cudnn
from torch.utils.data import DataLoader

from args import argument_parser
from torchreid import models
from torchreid.dataset_loader import ImageDataset
from torchreid.datasets import init_imgreid_dataset
from torchreid.datasets.collate_batch import val_collate_fn
from torchreid.transforms import build_transforms
from torchreid.utils.iotools import check_isfile
from torchreid.utils.reid_metric import R1_mAP, R1_mAP_reranking

max_tank = 50


def create_supervised_evaluator(model,
                                metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            # print(len(feat))
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        model,
        device,
        re_ranking,
        feat_norm,
        val_loader,
        num_query,
        dataset
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    evaluator = None
    if re_ranking == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP(num_query, dataset, max_rank=max_tank, feat_norm=feat_norm)},
                                                device=device)
    elif re_ranking == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP_reranking(num_query, dataset, max_rank=max_tank, feat_norm=feat_norm)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(re_ranking))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


parser = argument_parser()
args = parser.parse_args()

if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
        device = 'cuda'
    else:
        device = 'cpu'
        print("Currently using CPU, however, GPU is highly recommended")

    dataset = init_imgreid_dataset(args.source_names[0], root=args.root)

    transform_test_flip = build_transforms(args.height, args.width,
                                           is_train=False,
                                           data_augment=args.data_augment,
                                           flip=True)

    val_set = ImageDataset(dataset.query + dataset.gallery, transform_test_flip)
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            collate_fn=val_collate_fn)

    print(val_loader)

    model = models.init_model(name=args.arch,
                              num_classes=dataset.num_train_pids,
                              loss={'xent'},
                              use_gpu=use_gpu,
                              args=vars(args))

    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        try:
            checkpoint = torch.load(args.load_weights)
        except Exception as e:
            print(e)
            checkpoint = torch.load(args.load_weights, map_location={'cuda:0': 'cpu'})

        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))
    else:
        print("no weight file!!!")

    RE_RANKING = 'no'
    FEAT_NORM = 'yes'
    num_query = 0

    inference(model, device, RE_RANKING, FEAT_NORM, val_loader, num_query, dataset)