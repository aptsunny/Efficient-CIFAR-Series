"""
import os
import matplotlib.pyplot as plt
import numpy as np

def Read_Files(filename):
    X_axis = []  # X
    Y_axis = []  # Y
    with open(filename, 'r') as f:
        for line in f.readlines():
            x = line.split(" ")[0]
            y = line.split(" ")[1]
            X_axis.append(float(x))
            Y_axis.append(float(y))
    f.close()
    return X_axis, Y_axis

def plot_PF(X_axis, Y_axis):
    # 可以通过c 参数设置颜色
    # T = np.arctan2(Y_axis, X_axis)
    # plt.scatter(X_axis, Y_axis, s=2, c=T, alpha=0.5)
    plt.scatter(X_axis, Y_axis, s=20, alpha=0.5)
    plt.savefig(Figname + '.png', dpi=600)
    plt.show()

Filename = '../hpo_experiment/sample.txt'
Filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hpo_experiment/sample.txt')
Figname='CIHS1_hType_circle'

X_axis, Y_axis = Read_Files(Filename)
print(X_axis, Y_axis)
T = np.arctan2(Y_axis, X_axis)
plot_PF(X_axis, Y_axis)


#####################
# lr scheduler plot
#####################
"""
import torch
from torch.optim.lr_scheduler import _LRScheduler
class LinearLR(_LRScheduler):
    def __init__(self, optimizer, T, last_epoch=-1):
        self.T = float(T)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        rate = 1 - self.last_epoch / self.T
        return [rate * base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()
# from linearlr import LinearLR
from torch import optim
from warmup_scheduler import GradualWarmupScheduler

# it is recommended to update the lr per iteration 
# rather than per epoch, but both will work
max_iter = 100
base_lr = 1
dummy_optimizer = optim.SGD([torch.tensor(0)], base_lr)

# scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
scheduler_linear = LinearLR(dummy_optimizer, max_iter-6) # -6 means warm-up 5 epochs
scheduler = GradualWarmupScheduler(dummy_optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_linear)

for i in range(max_iter):
    # if pytorch < 1.1, update lr before training
    # scheduler.step()
    # do some training
    print('Iter %3d: lr: %g' % (i, scheduler.get_lr()[0]))
    # update scheduler (pytorch >= 1.1)
    scheduler.step()

import numpy as np
X=np.abs(np.random.rand(5, 2))
print(type(X[1]))
print(np.random.rand(5, 3))


print('\n')
print(np.random.rand(255, 2))
print((np.random.rand(255) + 1.5).astype(int))

print(np.random.randint(0, 255, (2, 3,)))


X=np.abs(np.random.rand(5, 2))

epoch_list = range(1,40)
for epoch in epoch_list:
    if epoch == 24 or epoch == 40:
        print(epoch)