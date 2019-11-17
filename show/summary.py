import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from show.logger import Logger


def show_model(model, model_name, input_size=(3, 256, 128)):
    # 模型打印
    # summary(model, input_size, device="cpu")
    # model可视化

    writer = SummaryWriter(os.path.join("..", "summary", model_name))
    x = torch.rand(1, input_size[0], input_size[1], input_size[2])
    writer.add_graph(model, x)
    writer.close()


if __name__ == '__main__':
    from torchreid import models
    from args import argument_parser

    parser = argument_parser()
    args = parser.parse_args()

    sys.stderr = sys.stdout = Logger("summary.txt")
    model = models.init_model(name=args.arch,
                              num_classes=4768,
                              loss={'xent'},
                              use_gpu=False,
                              args=vars(args))

    show_model(model, args.arch)
