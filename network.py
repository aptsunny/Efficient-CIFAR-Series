from core import *
from torch_backend import *

batch_norm = partial(BatchNorm, weight_init=None, bias_init=None)

def res_block(c_in, c_out, stride, **kw):
    block = {
        'bn1': batch_norm(c_in, **kw),
        'relu1': nn.ReLU(True),
        'branch': {
            'conv1': nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            'bn2': batch_norm(c_out, **kw),
            'relu2': nn.ReLU(True),
            'conv2': nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        }
    }
    projection = (stride != 1) or (c_in != c_out)
    if projection:
        block['conv3'] = (nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False), ['relu1'])
    block['add'] = (Add(), [('conv3' if projection else 'relu1'), 'branch/conv2'])
    return block

def DAWN_net(c=64, block=res_block, prep_bn_relu=False, concat_pool=True, **kw):
    if isinstance(c, int):
        c = [c, 2 * c, 4 * c, 4 * c]

    classifier_pool = {
        'in': Identity(),
        'maxpool': nn.MaxPool2d(4),
        'avgpool': (nn.AvgPool2d(4), ['in']),
        'concat': (Concat(), ['maxpool', 'avgpool']),
    } if concat_pool else {'pool': nn.MaxPool2d(4)}

    return {
        'input': (None, []),
        'prep': union({'conv': nn.Conv2d(3, c[0], kernel_size=3, stride=1, padding=1, bias=False)},
                      {'bn': batch_norm(c[0], **kw), 'relu': nn.ReLU(True)} if prep_bn_relu else {}),
        'layer1': {
            'block0': block(c[0], c[0], 1, **kw),
            'block1': block(c[0], c[0], 1, **kw),
        },
        'layer2': {
            'block0': block(c[0], c[1], 2, **kw),
            'block1': block(c[1], c[1], 1, **kw),
        },
        'layer3': {
            'block0': block(c[1], c[2], 2, **kw),
            'block1': block(c[2], c[2], 1, **kw),
        },
        'layer4': {
            'block0': block(c[2], c[3], 2, **kw),
            'block1': block(c[3], c[3], 1, **kw),
        },
        'final': union(classifier_pool, {
            'flatten': Flatten(),
            'linear': nn.Linear(2 * c[3] if concat_pool else c[3], 10, bias=True),
        }),
        'logits': Identity(),
    }

def conv_bn(c_in, c_out, ks=3, bn_weight_init=1.0, **kw):
    if isinstance(ks, list):
        basic = {}
        for ks_ in ks:
            basic[str(ks_)] =  {'conv': nn.Conv2d(c_in, c_out, kernel_size=ks_, stride=1, padding=ks_//2, bias=False),
                                'bn': batch_norm(c_out, **kw),
                                'relu': nn.ReLU(True)}
    else:
        basic = None

    return basic or {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=ks, stride=1, padding=ks//2, bias=False),
        'bn': batch_norm(c_out, **kw),
        'relu': nn.ReLU(True)
        # 'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        # 'bn': GhostBatchNorm(c_out, num_splits=2, **kw),
    }

def basic_net(ks, channels, weight, pool, num_classes, **kw):
    return {
        'input': (None, []),
        'prep': conv_bn(3, channels['prep'], ks=ks, bn_weight_init=1.0, **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], ks=ks, **kw), pool=pool), # 32->16
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], ks=ks, **kw), pool=pool), # 16->8
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], ks=ks, **kw), pool=pool), # 8->4
        'pool': nn.MaxPool2d(4),
        'flatten': Flatten(),
        'linear': nn.Linear(channels['layer3'], num_classes, bias=False),
        'logits': Mul(weight),
    }


def net(weight=0.125,
        channels=None,
        pool=nn.MaxPool2d(2),
        extra_layers={'prep':0, 'layer1':0, 'layer2':0, 'layer3':0},
        res_layers  ={'prep':0, 'layer1':0, 'layer2':0, 'layer3':0},
        num_classes=10,
        ks=3,
        **kw):

    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    layer2assignment = {'prep': '32', 'layer1':'16', 'layer2':'8', 'layer3':'4'} # divided by pooling
    assignment = {'32': 2, '16': 1, '8': 1, '4': 0}  # pooling+ maxpooling

    # defined residual: in+(res1+res2+relu)
    residual = lambda c, **kw: {'in': Identity(),
                                'res1': conv_bn(c, c, **kw),
                                'res2': conv_bn(c, c, **kw),
                                'add': (Add(), ['in', 'res2/relu'])}

    n = basic_net(ks, channels, weight, pool, num_classes, **kw)

    for layer in res_layers: # residual
        times = res_layers[layer]
        if times==0:
            continue
        for i in range(times):
            n[layer]['residual{}'.format(i+1)] = residual(channels[layer], **kw)
            assignment[layer2assignment[layer]]+=2

    for layer in extra_layers: # extra
        times = extra_layers[layer]
        if times==0:
            continue
        for i in range(times):
            n[layer]['extra{}'.format(i+1)] = conv_bn(channels[layer], channels[layer], **kw)
            assignment[layer2assignment[layer]]+=1

    return n, assignment


