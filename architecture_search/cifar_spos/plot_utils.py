#### Visdom

from visdom import Visdom
import numpy as np

import torch

"""
class AverageMeter(object):
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
"""

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=8887)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')


    def plot_bar(self, var_name, split_name, title_name, epoch, mutator):
        self.plots[var_name] = self.viz.bar(
            X=np.array([ # (5, 2)
                        mutator['conv1'],
                        mutator['conv2'],
                        mutator['conv3'],
                        mutator['conv4'],
                        mutator['skip']]),
            opts=dict(
                stacked=True,
                legend=['choice1', 'choice2'],
                rownames=['conv1', 'conv2', 'conv3', 'conv4', 'skip'],
                title=title_name,
                xlabel='conv',
                ylabel=var_name,
                env=self.env
            )
        )

    def plot_scatter(self, var_name, split_name, title_name, epoch, mutator):
        colors = np.random.randint(0, 255, (5, 3,)) #
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.scatter(
                # X=np.random.rand(255, 2), # cord
                # Y=(np.random.rand(255) + 1.5).astype(int), # class 1 or 2
                # ccc = torch.max(mutator, 0)[1]

                # [choice， nums of choices]
                X = np.array([
                    [1, self.ccc['conv1']],
                    [2, self.ccc['conv2']],
                    [3, self.ccc['conv3']],
                    [4, self.ccc['conv4']],
                    [5, self.ccc['skip']]
                ]),
                Y = np.array([1, 2, 3, 4, 5]),

                opts=dict(
                    markersize=10,
                    markercolor=colors,
                    legend=['conv1', 'conv2', 'conv3', 'conv4', 'skip'],
                    title=title_name,
                    xlabel='conv',
                    ylabel=var_name
                ),
            )
        else:
            # self.viz.scatter(
            #     X=np.random.rand(255),
            #     Y=np.random.rand(255),
            #     opts=dict(
            #         markersize=10,
            #         markercolor=colors[0].reshape(-1, 3),
            #
            #     ),
            #     name='1',
            #     update='append',
            #     win=self.plots[var_name])
            # print('mutator: ', mutator['conv1'])

            self.viz.scatter(
                # X=np.random.rand(255, 2),
                # Y=(np.random.rand(255) + 1.5).astype(int),
                X=np.array([
                    [1, mutator['conv1'][0]],
                    [2, mutator['conv2'][0]],
                    [3, mutator['conv3'][0]],
                    [4, mutator['conv4'][0]],
                    [5, mutator['skip'][0]]
                ]),
                Y=np.array([1, 2, 3, 4, 5]),

                opts=dict(
                    markersize=10,
                    markercolor=colors,
                ),
                update='append',
                win=self.plots[var_name])
