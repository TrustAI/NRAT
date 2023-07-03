# NRAT
Code for NRAT: Towards Adversarial Training with Inherent Label Noise
# Requisite
Python 3.6+  
Pytorch 1.8.0  
Torchvision 0.9.0  
Pandas 1.2.4  
# Usage
NRAT+symmetric label noise for CIFAR10:  
python train.py --log-dir 'trained_models' --desc 'NRAT_sym0.2' --lr 0.05 --beta 6.0 --NRAT --noise_rate 0.2  
NRAT+asymmetric label noise for CIFAR10:  
python train.py --log-dir 'trained_models' --desc 'NRAT_sym0.2' --lr 0.05 --beta 6.0 --asym --NRAT --noise_rate 0.2  
# Reference Code
[1] APL Loss: https://github.com/HanxunH/Active-Passive-Losses  
[2] TRADES: https://github.com/yaodongyu/TRADES/  
[3] DM-AT: https://github.com/wzekai99/DM-Improves-AT  

