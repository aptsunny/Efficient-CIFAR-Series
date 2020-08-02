import os
import matplotlib.pyplot as plt
import numpy as np

"""
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


Filename = './hpo_experiment/sample.txt'
Figname='CIHS1_hType_circle'
X_axis, Y_axis = Read_Files(Filename)
print(X_axis, Y_axis)
T = np.arctan2(Y_axis, X_axis)
plot_PF(X_axis, Y_axis)
"""
