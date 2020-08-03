import nni
import logging
_logger = logging.getLogger("cifar10_pytorch_automl")

from core import *
from torch_backend import *
from network import net


def train(model, lr_schedule, train_set, test_set, base_wd, batch_size, num_workers=0):

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

    # learning rate
    train_batches = DataLoader(train_set, batch_size, shuffle=True, set_random_choices=True, num_workers=num_workers)
    test_batches = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)
    lr = lambda step: lr_schedule(step / len(train_batches)) / batch_size

    # original
    opts = [SGD(trainable_params(model).values(),
                {'lr': lr,
                 'weight_decay': Const(base_wd * batch_size),
                 'momentum': Const(0.9),
                 })]

    # State need update by iteration
    logs, state = Table(), {MODEL: model, LOSS: x_ent_loss, OPTS: opts}

    best_acc=0.0
    actual_time = 0.0
    train_time = 0.0

    for epoch in range(lr_schedule.knots[-1]):
        result = train_epoch(state, Timer(torch.cuda.synchronize), train_batches, test_batches)
        actual_time = actual_time + result['total time']
        train_time = train_time + result['train time']
        logs.append(union({'epoch': epoch + 1, 'lr': lr_schedule(epoch + 1)}, result))

        nni.report_intermediate_result(result['valid']['acc'])

        if result['valid']['acc'] > best_acc:
            best_acc = result['valid']['acc']


    return logs, best_acc, actual_time, train_time

if __name__ == '__main__':
    try:
        # RCV_CONFIG = {}
        RCV_CONFIG = nni.get_next_parameter()
        _logger.debug(RCV_CONFIG)

        task = 'cifar100'
        # search space
        base_wd = RCV_CONFIG['base_wd'] if 'base_wd' in RCV_CONFIG else 5e-4
        logits_weight = RCV_CONFIG['logits_weight'] if 'logits_weight' in RCV_CONFIG else 0.125
        peak_epoch = RCV_CONFIG['peak_epoch'] if 'peak_epoch' in RCV_CONFIG else 5
        cutout_size = RCV_CONFIG['cutout'] if 'cutout' in RCV_CONFIG else 8
        total_epoch = RCV_CONFIG['total_epoch'] if 'total_epoch' in RCV_CONFIG else 24
        peak_lr = RCV_CONFIG['peak_lr'] if 'peak_lr' in RCV_CONFIG else 0.4

        channels = {'prep': RCV_CONFIG['prep'], 'layer1': RCV_CONFIG['layer1'], 'layer2': RCV_CONFIG['layer2'], 'layer3': RCV_CONFIG['layer3']} if 'prep' in RCV_CONFIG \
            else {'prep': 48, 'layer1': 112, 'layer2': 256, 'layer3': 384}

        extra_layers = {'prep': RCV_CONFIG['extra_prep'], 'layer1': RCV_CONFIG['extra_layer1'], 'layer2': RCV_CONFIG['extra_layer2'], 'layer3': RCV_CONFIG['extra_layer3']} if 'extra_prep' in RCV_CONFIG \
            else {'prep': 0, 'layer1': 0, 'layer2': 0, 'layer3': 0}

        res_layers = {'prep': RCV_CONFIG['res_prep'], 'layer1': RCV_CONFIG['res_layer1'], 'layer2': RCV_CONFIG['res_layer2'], 'layer3': RCV_CONFIG['res_layer3']} if 'res_prep' in RCV_CONFIG \
            else {'prep': 0, 'layer1': 1, 'layer2': 0, 'layer3': 1}

        config = [channels, extra_layers, res_layers]


        # dataset
        if task == 'cifar100':
            DATA_DIR = './data_cifar100'
            classes= 100
            dataset = cifar100(DATA_DIR)
        else:
            DATA_DIR = './data'
            classes = 10
            dataset = cifar10(DATA_DIR)
        timer = Timer()
        print('Preprocessing training data')
        transforms = [
            partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
            partial(transpose, source='NHWC', target='NCHW'),
        ]
        train_set = list(zip(*preprocess(dataset['train'], [partial(pad, border=4)] + transforms).values()))
        # Set_random_choices
        train_set_x = Transform(train_set, [Crop(32, 32), FlipLR(), Cutout(cutout_size, cutout_size)])
        print(f'Finished in {timer():.2} seconds')
        print('Preprocessing test data')
        test_set = list(zip(*preprocess(dataset['valid'], transforms).values()))
        print(f'Finished in {timer():.2} seconds')


        # Design search space
        n, assignment = net(weight=logits_weight,
                channels=channels,
                extra_layers=extra_layers,
                res_layers=res_layers,
                num_classes=classes)

        # selected path
        model = Network(n).to(device).half() # graph building

        # remove identity_nodes compare before
        remove_identity_nodes = lambda net: remove_by_type(net, Identity)
        colors = ColorMap()
        draw = lambda graph: DotGraph(
            {p: ({'fillcolor': colors[type(v)], 'tooltip': repr(v)}, inputs) for p, (v, inputs) in graph.items() if
             v is not None})
        draw(remove_identity_nodes(n))


        # train pipeline
        lr_schedule = PiecewiseLinear([0, peak_epoch, total_epoch], [0, peak_lr, 0])
        summary, best_acc, total_time, train_time = train(model, lr_schedule, train_set_x, test_set, base_wd, batch_size=512, num_workers=0)
        record(summary, task, tag='_'.join([str(assignment[i]) for i in assignment]), acc=float(best_acc), epoch=total_epoch, time=total_time, config=config)
        print('Finished Train/Valid in {:.2f} seconds, actual training time in {:.2f} seconds'.format(total_time, train_time))
        nni.report_final_result(best_acc)

    except Exception as exception:
        _logger.exception(exception)
        raise