import torch

from .resnet import Normalization
#from .preact_resnet import preact_resnet
from .resnet import resnet
#from .wideresnet import wideresnet

#from .preact_resnetwithswish import preact_resnetwithswish
#from .wideresnetwithswish import wideresnetwithswish

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101'
          ]


def create_model(name, normalize, info, device):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    
    if info['data'] in DATASETS and info['data'] not in ['tiny-imagenet']:

        if 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)

        else:
            raise ValueError('Invalid model name {}!'.format(name))
    
    else:
        raise ValueError('Models for {} not yet supported!'.format(info['data']))
        
    if normalize:
        model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
    else:
        model = torch.nn.Sequential(backbone)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model
