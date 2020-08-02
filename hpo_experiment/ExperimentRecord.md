

# 引言

一组优良的超参数可以极大程度上提升网络训练，除了本身超参数搜索的算法外(如何得到下一组超参数组合)，对网络进行高效训练也是十分必要的，这里提到如何用单独的GPU，在CIFAR-10/CIFAR-100 图像分类数据集上高效地训练残差网络（Residual networks）, 这对网络结构搜索以及超参数优化的效率提升有巨大好处。

# Content

# [Baseline](../notebook/demo.ipynb)

## Network visualisation

```python
        remove_identity_nodes = lambda net: remove_by_type(net, Identity)
        colors = ColorMap()
        draw = lambda graph: DotGraph(
            {p: ({'fillcolor': colors[type(v)], 'tooltip': repr(v)}, inputs) for p, (v, inputs) in graph.items() if
             v is not None})
        draw(build_graph(n))
```


# 

# Architecture Search

有问题，有其他程序在跑

test train time fp32
[16, 32, 48, 96] * 3 epoch = 5.3e+01
[16, 32, 48, 96] * 5 epoch = 5.9e+01
[16, 32, 48, 96] * 7 epoch = 1.1e+02

old fp16
[16, 32, 48, 96] * 5 epoch = 6e+01
[16, 32, 48, 96] * 7 epoch = 1.2e+02

new
[16, 32, 48, 96] * 5 epoch = 5.2e+01
[16, 32, 48, 96] * 7 epoch = 7.1e+01





# 实验记录

## baseline

```python
        n = net(weight=logits_weight,
                channels=channels,
                extra_layers=('layer1', 'layer3',),
                # res_layers=('layer1', 'layer3'),
                ks=3, # [3, 5],
                num_classes=classes)
```

```text
Network(
  (prep_conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (prep_bn): BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (prep_relu): ReLU(inplace=True)
  (layer1_conv): Conv2d(64, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (layer1_bn): BatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1_relu): ReLU(inplace=True)
  (layer1_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (layer1_extra_conv): Conv2d(112, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (layer1_extra_bn): BatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1_extra_relu): ReLU(inplace=True)
  (layer1_extra_1_conv): Conv2d(112, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (layer1_extra_1_bn): BatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1_extra_1_relu): ReLU(inplace=True)
  (layer2_conv): Conv2d(112, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (layer2_bn): BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer2_relu): ReLU(inplace=True)
  (layer2_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (layer3_conv): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (layer3_bn): BatchNorm(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer3_relu): ReLU(inplace=True)
  (layer3_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (layer3_extra_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (layer3_extra_bn): BatchNorm(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer3_extra_relu): ReLU(inplace=True)
  (layer3_extra_1_conv): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (layer3_extra_1_bn): BatchNorm(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer3_extra_1_relu): ReLU(inplace=True)
  (pool): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten()
  (linear): Linear(in_features=384, out_features=100, bias=False)
  (logits): Mul()
)
```

```text
       epoch           lr   train time   train loss    train acc   valid time   valid loss    valid acc   epoch time
           1       0.0800       5.4308       4.3200       0.0666       0.4296       3.8676       0.1033       5.8605
           2       0.1600       3.8228       3.4148       0.1801       0.2146       3.1490       0.2142       4.0374
           3       0.2400       3.6557       2.7415       0.2983       0.2078       2.4465       0.3502       3.8636
           4       0.3200       3.5254       2.3293       0.3807       0.2084       2.5319       0.3470       3.7338
           5       0.4000       3.3378       2.0901       0.4350       0.2115       2.1002       0.4358       3.5493
           6       0.3789       3.2168       1.8867       0.4825       0.2845       2.0987       0.4330       3.5014
           7       0.3579       3.3421       1.7172       0.5248       0.2246       1.9047       0.4885       3.5667
           8       0.3368       3.8336       1.5958       0.5544       0.2160       1.7638       0.5165       4.0496
           9       0.3158       3.7607       1.4897       0.5822       0.2056       2.0610       0.4658       3.9663
          10       0.2947       3.5481       1.4178       0.6033       0.1956       1.5872       0.5565       3.7437
          11       0.2737       3.7660       1.3460       0.6197       0.2092       1.5229       0.5753       3.9752
          12       0.2526       3.6411       1.2718       0.6387       0.2188       1.4178       0.6047       3.8599
          13       0.2316       3.7753       1.2046       0.6577       0.1931       1.6000       0.5671       3.9685
          14       0.2105       3.3390       1.1458       0.6747       0.2506       1.4272       0.5999       3.5896
          15       0.1895       3.7230       1.0816       0.6916       0.2084       1.3414       0.6155       3.9314
          16       0.1684       3.7872       1.0272       0.7064       0.2164       1.4668       0.5949       4.0036
          17       0.1474       3.6444       0.9623       0.7257       0.2084       1.3790       0.6215       3.8528
          18       0.1263       3.7156       0.8918       0.7449       0.2574       1.2262       0.6478       3.9730
          19       0.1053       3.5679       0.8218       0.7671       0.2128       1.2067       0.6609       3.7807
          20       0.0842       3.5874       0.7543       0.7838       0.2128       1.1597       0.6762       3.8002
          21       0.0632       3.4981       0.6776       0.8073       0.2233       1.0803       0.6965       3.7214
          22       0.0421       3.1418       0.5971       0.8316       0.1936       1.0140       0.7089       3.3354
          23       0.0211       3.3943       0.5162       0.8577       0.2057       0.9453       0.7278       3.5999
          24       0.0000       3.5504       0.4518       0.8790       0.2108       0.9004       0.7402       3.7612
Finished Train/Valid in 93.03 seconds
```

```text


```




## layer-wise lr

```text
假如lr_instead 则认为 lr即为lr_scale_ratio
'lr_instead': Const([0.1]*7+ [0.2]*7+ [0.3]*7+ [0.4]*7), #  后边几层学习率大，效果会好点，由于参数初始化地足够好

```

## Supernet adjustment

一次只取一个：
替换/ 
前边权重复用：
深度/ 

```python
            # re1
            if k.startswith('prep/extra'):
                continue
            elif k=='layer1/conv':
                ins = ['prep/relu']
                
            # re2
            # 超过固定层 extra
            if weight_id<6 and weight_id>2:
                weight_id = weight_id + 1
                continue
             
            # re4    
            if w.requires_grad:
                # re4
                # update(w.data, w.grad.data, v, **param_values)
                update(weight_id, w.data, w.grad.data, v, **param_values)             
```

## channel
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}

## depth
        # extra_layers=('prep', 'layer1', 'layer2' ,'layer3'), # 0:3.0; 1,3:3.4; 1,2,3:3.6; p,1,2,3:4.0->94.15 ;
        
        
        # n[layer]['extra_1'] = conv_bn(channels[layer], channels[layer], **kw) # (p,1,2,3)*2:5.0 -> 93.83 ;
        # n[layer]['extra_2'] = conv_bn(channels[layer], channels[layer], **kw)  # (p,1,2,3)*3:6.1 -> 93.44 ;
        # n[layer]['extra_3'] = conv_bn(channels[layer], channels[layer], **kw)  # (p,1,2,3)*3:7.1 -> 92.78 ;

## shortcut

        # res_layers=('prep', 'layer2'), # 'layer1', 'layer3',7.1 'prep', 'layer2'7.6



## default value
```text
        peak_lr = RCV_CONFIG['peak_lr']
        base_wd = RCV_CONFIG['base_wd']
        logits_weight = RCV_CONFIG['logits_weight']
        peak_epoch = RCV_CONFIG['peak_epoch']
        cutout_size = RCV_CONFIG['cutout']
        total_epoch = RCV_CONFIG['total_epoch']
        
        c_prep = RCV_CONFIG['prep']
        c_layer1 = RCV_CONFIG['layer1']
        c_layer2 = RCV_CONFIG['layer2']
        c_layer3 = RCV_CONFIG['layer3']
        channels = {'prep': c_prep, 'layer1': c_layer1, 'layer2': c_layer2, 'layer3': c_layer3}
        
        RCV_CONFIG = {'peak_lr': 0.4,
                      'base_wd': 5e-4,
                      'logits_weight': 0.125,
                      'peak_epoch': 5,
                      'cutout': 8}
                      
        RCV_CONFIG = {'peak_lr': 0.4,
              'prep': 64,
              'layer1': 128,
              'layer2': 256,
              'layer3': 512}


        # logits_weight = 0.125
        # peak_epoch = 5
        # cutout_size = 8
        # total_epoch = 24
        # batch_size = 512
        # peak_lr = 0.4
        # channels = {'prep': 64, 'layer1': 112, 'layer2': 256, 'layer3': 384}
   
   
        channels = {'prep': RCV_CONFIG['prep'], 'layer1': RCV_CONFIG['layer1'], 'layer2': RCV_CONFIG['layer2'], 'layer3': RCV_CONFIG['layer3']}
        peak_lr = RCV_CONFIG['peak_lr']  
        
    "peak_lr":{"_type": "loguniform", "_value": [6e-2, 6e-1]},
    "peak_epoch":{"_type": "randint", "_value": [3, 10]},
    "logits_weight":{"_type": "quniform", "_value": [0.1, 0.5, 0.05]},    
    "prep":{"_type": "choice", "_value": [16, 32, 48, 64]},
    "layer1":{"_type": "choice", "_value": [28, 56, 84, 112]},
    "layer2":{"_type": "choice", "_value": [64, 128, 192, 256]},
    "layer3":{"_type": "choice", "_value": [96, 192, 288, 384]}         
```

## operations
```text
[('input', (None, [])), 
 ('prep/conv', (Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('prep/bn', (BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('prep/relu', (ReLU(inplace=True), [-1])), 
 
 ('layer1/conv', (Conv2d(64, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('layer1/bn', (BatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('layer1/relu', (ReLU(inplace=True), [-1])), 
 ('layer1/pool', (MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), [-1])), 
 ('layer1/residual/in', (Identity(), [-1])), 
 ('layer1/residual/res1/conv', (Conv2d(112, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('layer1/residual/res1/bn', (BatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('layer1/residual/res1/relu', (ReLU(inplace=True), [-1])), 
 ('layer1/residual/res2/conv', (Conv2d(112, 112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('layer1/residual/res2/bn', (BatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('layer1/residual/res2/relu', (ReLU(inplace=True), [-1])), 
 ('layer1/residual/add', (Add(), ['in', 'res2/relu'])), 
 
 ('layer2/conv', (Conv2d(112, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('layer2/bn', (BatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('layer2/relu', (ReLU(inplace=True), [-1])), 
 ('layer2/pool', (MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), [-1])), 
 
 ('layer3/conv', (Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('layer3/bn', (BatchNorm(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('layer3/relu', (ReLU(inplace=True), [-1])), 
 ('layer3/pool', (MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), [-1])), 
 ('layer3/residual/in', (Identity(), [-1])), 
 ('layer3/residual/res1/conv', (Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('layer3/residual/res1/bn', (BatchNorm(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('layer3/residual/res1/relu', (ReLU(inplace=True), [-1])), 
 ('layer3/residual/res2/conv', (Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), [-1])), 
 ('layer3/residual/res2/bn', (BatchNorm(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), [-1])), 
 ('layer3/residual/res2/relu', (ReLU(inplace=True), [-1])), 
 ('layer3/residual/add', (Add(), ['in', 'res2/relu'])), 
 
 ('pool', (MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False), [-1])), 
 ('flatten', (Flatten(), [-1])), 
 ('linear', (Linear(in_features=384, out_features=100, bias=False), [-1])), 
 ('logits', (Mul(), [-1]))]
```


### Cifar10 Benchmark

original:(64,128,256,512)
```text
       epoch           lr   train time   train loss    train acc   valid time   valid loss    valid acc   epoch time
           1       0.0800       5.2681       1.6562       0.4016       0.4455       1.5635       0.4906       5.7136
           2       0.1600       3.1410       0.9657       0.6553       0.2137       0.8271       0.7128       3.3547
           3       0.2400       3.1374       0.7293       0.7461       0.2134       0.7769       0.7348       3.3508
           4       0.3200       3.1332       0.6293       0.7814       0.2134       0.6103       0.7904       3.3467
           5       0.4000       3.1327       0.5600       0.8067       0.2133       0.9152       0.7023       3.3459
           6       0.3789       3.1384       0.4948       0.8286       0.2132       0.6222       0.7849       3.3516
           7       0.3579       3.1306       0.4446       0.8497       0.2133       0.4870       0.8347       3.3439
           8       0.3368       3.1312       0.4151       0.8578       0.2153       0.5974       0.7904       3.3465
           9       0.3158       3.1370       0.3826       0.8688       0.2131       0.5073       0.8247       3.3500
          10       0.2947       3.1295       0.3662       0.8751       0.2132       0.6663       0.7728       3.3427
          11       0.2737       3.1298       0.3445       0.8833       0.2130       0.5169       0.8246       3.3428
          12       0.2526       3.1364       0.3264       0.8892       0.2131       0.5160       0.8269       3.3496
          13       0.2316       3.1337       0.3081       0.8946       0.2161       0.3941       0.8704       3.3498
          14       0.2105       3.1313       0.2868       0.9036       0.2270       0.4128       0.8638       3.3582
          15       0.1895       3.1310       0.2664       0.9102       0.2132       0.3848       0.8683       3.3441
          16       0.1684       3.1347       0.2534       0.9143       0.2153       0.5107       0.8349       3.3501
          17       0.1474       3.1407       0.2312       0.9226       0.2132       0.3146       0.8941       3.3540
          18       0.1263       3.1346       0.2123       0.9293       0.2132       0.5129       0.8316       3.3479
          19       0.1053       3.1354       0.1867       0.9383       0.2135       0.3226       0.8946       3.3489
          20       0.0842       3.1416       0.1647       0.9462       0.2136       0.2554       0.9138       3.3552
          21       0.0632       3.1368       0.1398       0.9544       0.2132       0.2517       0.9191       3.3501
          22       0.0421       3.1345       0.1165       0.9636       0.2133       0.2133       0.9293       3.3478
          23       0.0211       3.1435       0.0939       0.9716       0.2134       0.1974       0.9353       3.3569
          24       0.0000       3.1389       0.0759       0.9782       0.2134       0.1864       0.9380       3.3524
Finished Train/Valid in 82.75 seconds
```

original:(64,112,256,384)
```text
       epoch           lr   train time   train loss    train acc   valid time   valid loss    valid acc   epoch time
           1       0.0800       4.6210       1.6546       0.4033       0.3592       1.3337       0.5179       4.9802
           2       0.1600       2.8338       0.9414       0.6660       0.1875       0.8578       0.7109       3.0213
           3       0.2400       2.8343       0.7329       0.7454       0.1868       0.6906       0.7616       3.0212
           4       0.3200       2.8266       0.6206       0.7831       0.1870       0.6581       0.7755       3.0136
           5       0.4000       2.8262       0.5654       0.8023       0.1866       0.6772       0.7712       3.0128
           6       0.3789       2.8363       0.5065       0.8261       0.1866       0.6560       0.7669       3.0228
           7       0.3579       2.8254       0.4580       0.8433       0.1867       0.5341       0.8169       3.0120
           8       0.3368       2.8260       0.4215       0.8558       0.1868       0.4575       0.8443       3.0128
           9       0.3158       2.8350       0.3926       0.8668       0.1865       0.5829       0.8074       3.0215
          10       0.2947       2.8275       0.3773       0.8704       0.1889       0.5045       0.8248       3.0164
          11       0.2737       2.8260       0.3531       0.8804       0.1868       0.4697       0.8371       3.0127
          12       0.2526       2.8358       0.3357       0.8862       0.1865       0.5195       0.8297       3.0223
          13       0.2316       2.8289       0.3192       0.8920       0.1869       0.4282       0.8549       3.0158
          14       0.2105       2.8311       0.3030       0.8962       0.2049       0.4699       0.8406       3.0360
          15       0.1895       2.8421       0.2827       0.9037       0.1869       0.4603       0.8382       3.0290
          16       0.1684       2.8278       0.2618       0.9117       0.1871       0.3078       0.8936       3.0149
          17       0.1474       2.8394       0.2437       0.9167       0.1868       0.3591       0.8801       3.0262
          18       0.1263       2.8280       0.2195       0.9260       0.1868       0.3329       0.8863       3.0148
          19       0.1053       2.8289       0.2021       0.9311       0.1865       0.2878       0.9003       3.0154
          20       0.0842       2.8411       0.1788       0.9412       0.1889       0.3066       0.8984       3.0300
          21       0.0632       2.8328       0.1577       0.9483       0.1888       0.2620       0.9155       3.0216
          22       0.0421       2.8290       0.1282       0.9598       0.1866       0.2069       0.9297       3.0156
          23       0.0211       2.8379       0.1080       0.9668       0.1870       0.2044       0.9323       3.0250
          24       0.0000       2.8290       0.0854       0.9750       0.1872       0.1842       0.9372       3.0162
Finished Train/Valid in 74.43 seconds
```


### Cifar100 Benchmark

original:(64,128,256,512)
```text
       epoch           lr   train time   train loss    train acc   valid time   valid loss    valid acc   epoch time
           1       0.0800       5.2754       4.2068       0.0722       0.4376       3.6977       0.1333       5.7130
           2       0.1600       3.1515       3.2299       0.2134       0.2142       2.8563       0.2783       3.3657
           3       0.2400       3.1552       2.5306       0.3441       0.2137       2.2743       0.4020       3.3689
           4       0.3200       3.1397       2.1229       0.4333       0.2137       2.1333       0.4321       3.3534
           5       0.4000       3.1374       1.8773       0.4886       0.2135       2.1589       0.4374       3.3510
           6       0.3789       3.1440       1.7048       0.5310       0.2138       1.8141       0.5031       3.3577
           7       0.3579       3.1349       1.5363       0.5740       0.2138       1.7351       0.5272       3.3487
           8       0.3368       3.1369       1.4208       0.6022       0.2133       1.6288       0.5532       3.3502
           9       0.3158       3.1429       1.3322       0.6270       0.2134       1.5877       0.5596       3.3563
          10       0.2947       3.1350       1.2445       0.6508       0.2134       1.5829       0.5625       3.3483
          11       0.2737       3.1376       1.1831       0.6696       0.2134       1.6486       0.5594       3.3510
          12       0.2526       3.1446       1.1200       0.6850       0.2134       1.5578       0.5777       3.3580
          13       0.2316       3.1390       1.0544       0.6990       0.2134       1.3999       0.6125       3.3524
          14       0.2105       3.1391       0.9878       0.7204       0.2270       1.3671       0.6179       3.3661
          15       0.1895       3.1416       0.9444       0.7328       0.2162       1.3801       0.6198       3.3578
          16       0.1684       3.1520       0.8743       0.7525       0.2136       1.2706       0.6414       3.3655
          17       0.1474       3.1456       0.8044       0.7711       0.2137       1.2222       0.6626       3.3593
          18       0.1263       3.1372       0.7411       0.7904       0.2134       1.1606       0.6783       3.3506
          19       0.1053       3.1392       0.6706       0.8122       0.2134       1.1621       0.6753       3.3526
          20       0.0842       3.1450       0.6001       0.8340       0.2162       1.0832       0.7004       3.3612
          21       0.0632       3.1410       0.5237       0.8564       0.2161       0.9917       0.7238       3.3571
          22       0.0421       3.1385       0.4530       0.8787       0.2134       0.9506       0.7346       3.3519
          23       0.0211       3.1463       0.3747       0.9038       0.2134       0.8988       0.7448       3.3597
          24       0.0000       3.1406       0.3131       0.9247       0.2136       0.8649       0.7582       3.3542
Finished Train/Valid in 82.91 seconds
```

original:(64,112,256,384)
```text
       epoch           lr   train time   train loss    train acc   valid time   valid loss    valid acc   epoch time
           1       0.0800       4.6238       4.2513       0.0725       0.3769       3.7727       0.1163       5.0007
           2       0.1600       2.8339       3.2977       0.2017       0.1869       2.9349       0.2556       3.0208
           3       0.2400       2.8348       2.6009       0.3281       0.1870       2.7465       0.3088       3.0218
           4       0.3200       2.8262       2.1780       0.4206       0.1868       2.2030       0.4044       3.0130
           5       0.4000       2.8263       1.9388       0.4729       0.1871       1.8420       0.4964       3.0135
           6       0.3789       2.8373       1.7520       0.5222       0.1872       2.1163       0.4372       3.0244
           7       0.3579       2.8240       1.5811       0.5598       0.1867       1.7601       0.5161       3.0108
           8       0.3368       2.8239       1.4803       0.5866       0.1867       1.7218       0.5307       3.0106
           9       0.3158       2.8330       1.3878       0.6116       0.1867       1.7887       0.5149       3.0197
          10       0.2947       2.8253       1.3136       0.6329       0.1868       1.6937       0.5321       3.0121
          11       0.2737       2.8287       1.2531       0.6486       0.1866       1.4948       0.5910       3.0153
          12       0.2526       2.8328       1.1801       0.6669       0.1868       1.5391       0.5775       3.0196
          13       0.2316       2.8275       1.1250       0.6818       0.1867       1.4173       0.6018       3.0141
          14       0.2105       2.8270       1.0591       0.7005       0.2023       1.6151       0.5620       3.0293
          15       0.1895       2.8257       1.0004       0.7163       0.1870       1.3851       0.6194       3.0126
          16       0.1684       2.8280       0.9466       0.7317       0.1867       1.3412       0.6293       3.0147
          17       0.1474       2.8355       0.8793       0.7514       0.1867       1.2335       0.6554       3.0222
          18       0.1263       2.8319       0.8326       0.7633       0.1888       1.1905       0.6667       3.0207
          19       0.1053       2.8294       0.7559       0.7849       0.1868       1.1495       0.6770       3.0162
          20       0.0842       2.8458       0.6936       0.8034       0.1867       1.1408       0.6836       3.0325
          21       0.0632       2.8262       0.6176       0.8282       0.1870       1.0391       0.7032       3.0133
          22       0.0421       2.8278       0.5440       0.8500       0.1868       0.9933       0.7167       3.0147
          23       0.0211       2.8378       0.4688       0.8744       0.1892       0.9372       0.7333       3.0270
          24       0.0000       2.8410       0.4042       0.8955       0.1868       0.8943       0.7446       3.0279
Finished Train/Valid in 74.43 seconds
```


[apex 混合精度训练](https://github.com/NVIDIA/apex) 用于结构搜索部分
```bash
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```python
# Initialization
opt_level = 'O1'
model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# Train your model
...
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
...
``` 


## [快速入门](https://github.com/microsoft/nni/blob/master/docs/zh_CN/Tutorial/QuickStart.md)

## 安装

目前支持 Linux、macOS 和 Windows。 Ubuntu 16.04 或更高版本、macOS 10.14.1 和 Windows 10.1809 均经过测试并支持。 在 `python >= 3.5` 的环境中，只需要运行 `pip install` 即可完成安装。

### Linux 和 macOS

```bash
python3 -m pip install --upgrade nni
```

### Windows

```bash
python -m pip install --upgrade nni
```

```eval_rst
.. Note:: 在 Linux 和 macOS 上，如果要将 NNI 安装到当前用户的 home 目录中，可使用 ``--user``；这不需要特殊权限。
```

```eval_rst
.. Note:: 如果出现 ``Segmentation fault`` 这样的错误，参考 :doc:`常见问题 <FAQ>`。
```

```eval_rst
.. Note:: NNI 的系统需求，参考 :doc:`Linux 和 Mac <InstallationLinux>` 或 :doc:`Windows <InstallationWin>` 的安装教程。
```

## MNIST 上的 "Hello World"

NNI 是一个能进行自动机器学习实验的工具包。 它可以自动进行获取超参、运行 Trial，测试结果，调优超参的循环。 在这里，将演示如何使用 NNI 帮助找到 MNIST 模型的最佳超参数。

这是还**没有 NNI** 的示例代码，用 CNN 在 MNIST 数据集上训练：

```python
def run_trial(params):
    # 输入数据
    mnist = input_data.read_data_sets(params['data_dir'], one_hot=True)
    # 构建网络
    mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'],
                                 channel_2_num=params['channel_2_num'],
                                 conv_size=params['conv_size'],
                                 hidden_size=params['hidden_size'],
                                 pool_size=params['pool_size'],
                                 learning_rate=params['learning_rate'])
    mnist_network.build_network()

    test_acc = 0.0
    with tf.Session() as sess:
        # 训练网络
        mnist_network.train(sess, mnist)
        # 评估网络
        test_acc = mnist_network.evaluate(mnist)

if __name__ == '__main__':
    params = {'data_dir': '/tmp/tensorflow/mnist/input_data',
              'dropout_rate': 0.5,
              'channel_1_num': 32,
              'channel_2_num': 64,
              'conv_size': 5,
              'pool_size': 2,
              'hidden_size': 1024,
              'learning_rate': 1e-4,
              'batch_num': 2000,
              'batch_size': 32}
    run_trial(params)
```

完整实现请参考 [examples/trials/mnist-tfv1/mnist_before.py](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/mnist_before.py)

上面的代码一次只能尝试一组参数，如果想要调优学习率，需要手工改动超参，并一次次尝试。

NNI 用来帮助超参调优。它的流程如下：

```text
输入: 搜索空间, Trial 代码, 配置文件
输出: 一组最优的参数配置

1: For t = 0, 1, 2, ..., maxTrialNum,
2:      hyperparameter = 从搜索空间选择一组参数
3:      final result = run_trial_and_evaluate(hyperparameter)
4:      返回最终结果给 NNI
5:      If 时间达到上限,
6:          停止实验
7: 返回最好的实验结果
```

如果需要使用 NNI 来自动训练模型，找到最佳超参，需要根据代码，进行如下三步改动：

### 启动 Experiment 的三个步骤

**第一步**：编写 JSON 格式的`搜索空间`文件，包括所有需要搜索的超参的`名称`和`分布`（离散和连续值均可）。

```diff
-   params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64,
-   'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
+ {
+     "dropout_rate":{"_type":"uniform","_value":[0.5, 0.9]},
+     "conv_size":{"_type":"choice","_value":[2,3,5,7]},
+     "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
+     "batch_size": {"_type":"choice", "_value": [1, 4, 8, 16, 32]},
+     "learning_rate":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]}
+ }
```

*示例：[search_space.json](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/search_space.json)*

**第二步**：修改 `Trial` 代码来从 NNI 获取超参，并返回 NNI 最终结果。

```diff
+ import nni

  def run_trial(params):
      mnist = input_data.read_data_sets(params['data_dir'], one_hot=True)

      mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'], channel_2_num=params['channel_2_num'], conv_size=params['conv_size'], hidden_size=params['hidden_size'], pool_size=params['pool_size'], learning_rate=params['learning_rate'])
      mnist_network.build_network()

      with tf.Session() as sess:
          mnist_network.train(sess, mnist)
          test_acc = mnist_network.evaluate(mnist)

+         nni.report_final_result(test_acc)

  if __name__ == '__main__':

-     params = {'data_dir': '/tmp/tensorflow/mnist/input_data', 'dropout_rate': 0.5, 'channel_1_num': 32, 'channel_2_num': 64,
-     'conv_size': 5, 'pool_size': 2, 'hidden_size': 1024, 'learning_rate': 1e-4, 'batch_num': 2000, 'batch_size': 32}
+     params = nni.get_next_parameter()
      run_trial(params)
```

*示例：[mnist.py](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/mnist.py)*

**第三步**：定义 YAML 格式的`配置`文件，其中声明了搜索空间和 Trial 文件的`路径`。 它还提供其他信息，例如调整算法，最大 Trial 运行次数和最大持续时间的参数。

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
trainingServicePlatform: local
# 搜索空间文件
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
# 运行的命令，以及 Trial 代码的路径
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
```

```eval_rst
.. Note:: 如果要使用远程计算机或集群作为 :doc:`训练平台 <../TrainingService/Overview>`，为了避免产生过大的网络压力，NNI 限制了文件的最大数量为 2000，大小为 300 MB。 如果 codeDir 中包含了过多的文件，可添加 ``.nniignore`` 文件来排除部分，与 ``.gitignore`` 文件用法类似。 参考 `git documentation <https://git-scm.com/docs/gitignore#_pattern_format>` ，了解更多如何编写此文件的详细信息 _。
```

*示例: [config.yml](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/config.yml) [.nniignore](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1/.nniignore)*

上面的代码都已准备好，并保存在 [examples/trials/mnist-tfv1/](https://github.com/Microsoft/nni/tree/master/examples/trials/mnist-tfv1)。

#### Linux 和 macOS

从命令行使用 **config.yml** 文件启动 MNIST Experiment 。

```bash
nnictl create --config nni/examples/trials/mnist-tfv1/config.yml
```

#### Windows

从命令行使用 **config_windows.yml** 文件启动 MNIST Experiment 。

```bash
nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml
```

```eval_rst
.. Note:: 如果使用 Windows，则需要在 config.yml 文件中，将 ``python3`` 改为 ``python``，或者使用 config_windows.yml 来开始 Experiment。
```

```eval_rst
.. Note:: ``nnictl`` 是一个命令行工具，用来控制 NNI Experiment，如启动、停止、继续 Experiment，启动、停止 NNIBoard 等等。 点击 :doc:`这里 <Nnictl>` 查看 ``nnictl`` 的更多用法。
```

在命令行中等待输出 `INFO: Successfully started experiment!`。 此消息表明 Experiment 已成功启动。 期望的输出如下：

```text
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: [Your IP]:8080
-----------------------------------------------------------------------

You can use these commands to get more information about the experiment
-----------------------------------------------------------------------
         commands                       description

1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
-----------------------------------------------------------------------
```

如果根据上述步骤准备好了相应 `Trial`, `搜索空间`和`配置`，并成功创建的 NNI 任务。NNI 会自动开始通过配置的搜索空间来运行不同的超参集合，搜索最好的超参。 通过 Web 界面可看到 NNI 的进度。

## Web 界面

启动 Experiment 后，可以在命令行界面找到如下的 `Web 界面地址`：

```text
Web 地址为：[IP 地址]:8080
```

在浏览器中打开 `Web 界面地址`(即：`[IP 地址]:8080`)，就可以看到 Experiment 的详细信息，以及所有的 Trial 任务。 如果无法打开终端中的 Web 界面链接，可以参考[常见问题](FAQ.md)。

### 查看概要页面

点击 "Overview" 标签。

Experiment 相关信息会显示在界面上，配置和搜索空间等。 可通过 **Download** 按钮来下载信息和参数。 可以在 Experiment 运行时随时下载结果，也可以等到执行结束。

![](../../img/QuickStart1.png)

前 10 个 Trial 将列在 Overview 页上。 可以在 "Trials Detail" 页面上浏览所有 Trial。

![](../../img/QuickStart2.png)

### 查看 Trial 详情页面

点击 "Default Metric" 来查看所有 Trial 的点图。 悬停鼠标来查看默认指标和搜索空间信息。

![](../../img/QuickStart3.png)

点击 "Hyper Parameter" 标签查看图像。

* 可选择百分比查看最好的 Trial。
* 选择两个轴来交换位置。

![](../../img/QuickStart4.png)

点击 "Trial Duration" 标签来查看柱状图。

![](../../img/QuickStart5.png)

下面是所有 Trial 的状态。 包括：

* Trial 详情：Trial 的 id，持续时间，开始时间，结束时间，状态，精度和搜索空间文件。
* 如果在 OpenPAI 平台上运行，还可以看到 hdfsLog。
* Kill: 可结束在 `Running` 状态的任务。
* Support: 用于搜索某个指定的 Trial。

![](../../img/QuickStart6.png)

* 中间结果图

![](../../img/QuickStart7.png)

## 相关主题

* [尝试不同的 Tuner](../Tuner/BuiltinTuner.md)
* [尝试不同的 Assessor](../Assessor/BuiltinAssessor.md)
* [使用命令行工具 nnictl](Nnictl.md)
* [如何实现 Trial 代码](../TrialExample/Trials.md)
* [如何在本机运行 Experiment (支持多 GPU 卡)？](../TrainingService/LocalMode.md)
* [如何在多机上运行 Experiment？](../TrainingService/RemoteMachineMode.md)
* [如何在 OpenPAI 上运行 Experiment？](../TrainingService/PaiMode.md)
* [如何通过 Kubeflow 在 Kubernetes 上运行 Experiment？](../TrainingService/KubeflowMode.md)
* [如何通过 FrameworkController 在 Kubernetes 上运行 Experiment？](../TrainingService/FrameworkControllerMode.md)