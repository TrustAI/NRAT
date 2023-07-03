import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from .train import SCHEDULERS

from .utils import str2bool, str2float


def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description='Standard + Adversarial Training.')

    parser.add_argument('--augment', type=str, default='base', choices=['none', 'base', 'cutout', 'autoaugment', 'randaugment', 'idbh'], help='Augment training set.')

    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=256, help='Batch size for testing.')
    
    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')
    
    parser.add_argument('-d', '--data', type=str, default='cifar10', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str, required=True, 
                        help='Description of experiment. It will be used to name directories.')

    parser.add_argument('-m', '--model', choices=MODELS, default='resnet18', help='Model architecture to be used.')

    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('-na', '--num-adv-epochs', type=int, default=200, help='Number of adversarial training epochs.')
    parser.add_argument('--adv-eval-freq', type=int, default=1, help='Adversarial evaluation frequency (in epochs).')
    # test_adv_acc frequence
    parser.add_argument('--beta', default=None, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.')
    
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='step', help='Type of scheduler.')
   
    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--keep-clean', type=str2bool, default=False, help='Use clean samples during adversarial training.')

    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')

    parser.add_argument('--NRAT', action='store_true', default=False, help='NRAT training.')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    ### Consistency
    parser.add_argument('--consistency', action='store_true', default=False, help='use Consistency.')
    parser.add_argument('--cons_lambda', type=float, default=1.0, help='lambda for Consistency.')
    parser.add_argument('--cons_tem', type=float, default=0.5, help='temperature for Consistency.')

    ### Resume
    parser.add_argument('--resume_path', default='', type=str)
    
    # noisy labels
    parser.add_argument('--apl', action='store_true', default=False)
    parser.add_argument('--asym', action='store_true', default=False)
    parser.add_argument('--noise_rate', type=float, default=0)
    
    
    return parser


def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')
        
    parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')  #test怎么不是1w

    # eval-aa.py
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')

    # eval-adv.py
    parser.add_argument('--source', type=str, default=None, help='Path to source model for black-box evaluation.')
    parser.add_argument('--wb', action='store_true', default=False, help='Perform white-box PGD evaluation.')
    
    # eval-rb.py
    parser.add_argument('--threat', type=str, default='corruptions', choices=['corruptions', 'Linf', 'L2'],
                        help='Threat model for RobustBench evaluation.')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser

