import nni
import logging
_logger = logging.getLogger("cifar10_pytorch_automl")

from core import *
from torch_backend import *
from network import net

def tsv(logs):
    data = [(output['epoch'], output['epoch time']/3600, output['valid']['acc']*100) for output in logs]
    return '\n'.join(['epoch\thours\ttop1Accuracy']+[f'{epoch}\t{hours:.8f}\t{acc:.2f}' for (epoch, hours, acc) in data])


def train(model, lr_schedule, train_set, test_set, base_wd, batch_size, num_workers=0):

    train_batches = DataLoader(train_set, batch_size, shuffle=True, set_random_choices=True, num_workers=num_workers)
    test_batches = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)

    lr = lambda step: lr_schedule(step / len(train_batches)) / batch_size

    # original
    opts = [SGD(trainable_params(model).values(),
                {'lr': lr,
                 'weight_decay': Const(base_wd * batch_size),
                 'momentum': Const(0.9),
                 })]


    # layer-wise learning rate
    # split_list = [3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    # aaa = check_split_layer_wise_lr(model, split_list)
    """
    # conv_lr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    conv_lr = [0.4]*10
    opts = [SGD(trainable_params_(model).values(),
                {'lr': lr,
                 'weight_decay': Const(base_wd * batch_size),
                 'momentum': Const(0.9),
                 'lr_instead': Const(split_layer_wise_lr(split_list=split_list, conv_lr=conv_lr))
                })]
    """

    # state need update by iteration
    logs, state = Table(), {MODEL: model, LOSS: x_ent_loss, OPTS: opts}

    best_acc=0.0
    ture_time = 0.0

    for epoch in range(lr_schedule.knots[-1]):
        result = train_epoch(state, Timer(torch.cuda.synchronize), train_batches, test_batches)
        # print(result['valid']['acc'])
        ture_time = ture_time + result['epoch time']
        logs.append(union({'epoch': epoch + 1,
                           'lr': lr_schedule(epoch + 1)},
                           result))

        nni.report_intermediate_result(result['valid']['acc'])


        if result['valid']['acc'] > best_acc:
            best_acc = result['valid']['acc']

        # logs.append(union({'epoch': epoch + 1,
        #                    'lr': lr_schedule(epoch + 1)},
        #                   train_epoch(state, Timer(torch.cuda.synchronize), train_batches, test_batches)))


    return logs, best_acc, ture_time

if __name__ == '__main__':
    try:
        RCV_CONFIG = nni.get_next_parameter()
        _logger.debug(RCV_CONFIG)

        task = 'cifar100_dataset'

        base_wd = 5e-4
        logits_weight = 0.125
        peak_epoch = 5
        cutout_size = 8
        total_epoch = 24
        batch_size = 512


        # peak_lr = 0.4
        # channels = {'prep': 64, 'layer1': 112, 'layer2': 256, 'layer3': 384}
        # search space
        c_prep = RCV_CONFIG['prep']
        c_layer1 = RCV_CONFIG['layer1']
        c_layer2 = RCV_CONFIG['layer2']
        c_layer3 = RCV_CONFIG['layer3']
        channels = {'prep': c_prep, 'layer1': c_layer1, 'layer2': c_layer2, 'layer3': c_layer3}
        peak_lr = RCV_CONFIG['peak_lr']

        lr_schedule = PiecewiseLinear([0, peak_epoch, total_epoch], [0, peak_lr, 0])
        # dataset
        if task == 'cifar100_dataset':
            DATA_DIR = './data_cifar100'
            classes= 100
            dataset = cifar100(DATA_DIR)
        else:
            DATA_DIR = './data'
            classes = 10
            dataset = cifar10(DATA_DIR)

        # considerate training time
        timer = Timer()
        print('Preprocessing training data')
        transforms = [
            partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
            partial(transpose, source='NHWC', target='NCHW'),
        ]
        train_set = list(zip(*preprocess(dataset['train'],
                                         [partial(pad, border=4)] +
                                         transforms).values()))
        print(f'Finished in {timer():.2} seconds')

        print('Preprocessing test data')
        test_set = list(zip(*preprocess(dataset['valid'],
                                        transforms).values()))
        print(f'Finished in {timer():.2} seconds')

        # remove_identity_nodes = lambda net: remove_by_type(net, Identity)

        # Design search space
        n = net(weight=logits_weight,
                channels=channels,
                extra_layers=('layer1', 'layer3',),
                # res_layers=('layer1', 'layer3'),
                ks=3, # [3, 5],
                num_classes=classes)

        # colors = ColorMap()
        # draw = lambda graph: DotGraph(
        #     {p: ({'fillcolor': colors[type(v)], 'tooltip': repr(v)}, inputs) for p, (v, inputs) in graph.items() if
        #      v is not None})
        # draw(build_graph(n))

        # selected path
        model = Network(n).to(device).half() # graph building


        train_set_x = Transform(train_set, [Crop(32, 32), FlipLR(), Cutout(cutout_size, cutout_size)])

        # train pipeline
        summary, best_acc, total_time = train(model, lr_schedule, train_set_x, test_set, base_wd, batch_size=batch_size, num_workers=0)

        # import os
        # with open(os.path.join(os.path.expanduser('.'), 'cifar100_training_logs.tsv'), 'w') as f:
        #     f.write(tsv(summary.log))
        print('Finished Train/Valid in {:.2f} seconds'.format(total_time))

        nni.report_final_result(best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise