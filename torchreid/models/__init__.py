from __future__ import absolute_import

from . import densenet, resnet
from .hacnn import *
# from .densenet import *
from .inceptionresnetv2 import *
from .inceptionv4 import *
from .mlfn import *
from .mobilenetv2 import *
from .mudeep import *
from .nasnet import *
from .pcb import *
# from .resnet import *
from .resnetmid import *
from .resnext import *
from .senet import *
from .shufflenet import *
from .squeezenet import *
from .xception import *

__model_factory = {
    **densenet.model_mapping,
    'resnet50': resnet.resnet50,
    'resnet50_mgn_like': resnet.resnet50_mgn_like,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'inceptionresnetv2': InceptionResNetV2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2': MobileNetV2,
    'shufflenet': ShuffleNet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
