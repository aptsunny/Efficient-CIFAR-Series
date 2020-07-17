# Efficient-CIFAR-Series

## [NNI Experiment](https://github.com/microsoft/nni)

### Install
NNI supports and is tested on Ubuntu >= 16.04, macOS >= 10.14.1, and Windows 10 >= 1809. Simply run the following `pip install` in an environment that has `python 64-bit >= 3.5`.

Linux or macOS

```bash
python3 -m pip install --upgrade nni
```

### Start Experiment

XXXX>1024

```bash
nnictl create --config config.yml --port XXXX 
```

Search space configuration: 
Then edit `search_space.json`, you can add the hyper-parameters as follows:
```
+-----------------+------------+-------------------------------------------------------------------------------+
| Name            | default    |                        Hyper-parameters                                       |
+-----------------+------------+-------------------------------------------------------------------------------+
| peak_lr         | 0.4        |    "peak_lr":{"_type": "loguniform", "_value": [4e-5, 4e-1]},                 |
+-----------------+------------+-------------------------------------------------------------------------------+
| base_wd         | 5e-4       |    "base_wd":{"_type": "loguniform", "_value": [5e-5, 5e-3]},                 |
+-----------------+------------+-------------------------------------------------------------------------------+
| logits_weight   | 0.125      |    "logits_weight":{"_type":"choice", "_value":[0.0625, 0.125, 0.25, 0.5, 1]},|
+-----------------+------------+-------------------------------------------------------------------------------+
| peak_epoch      | 5          |    "peak_epoch":{"_type":"choice", "_value":[5, 10, 15, 20]},                 |
+-----------------+------------+-------------------------------------------------------------------------------+
| cutout          | 8          |    "cutout":{"_type":"choice", "_value":[10, 8, 6, 4]},                       |
+-----------------+------------+-------------------------------------------------------------------------------+
| total_epoch     | 24         |    "total_epoch":{"_type":"choice", "_value":[24, 32, 40]}                    |
+-----------------+------------+-------------------------------------------------------------------------------+
| prep            | 64         |    "prep":{"_type":"choice", "_value":[16, 32, 48, 64]},                      |
+-----------------+------------+-------------------------------------------------------------------------------+
| layer1          | 128        |    "layer1":{"_type":"choice", "_value":[64, 80, 96, 112, 128]},              |
+-----------------+------------+-------------------------------------------------------------------------------+
| layer2          | 256        |    "layer2":{"_type":"choice", "_value":[128, 160, 192, 224, 256]},           |
+-----------------+------------+-------------------------------------------------------------------------------+
| layer3          | 512        |    "layer3":{"_type":"choice", "_value":[256, 320, 384, 448, 512]}            |
+-----------------+------------+-------------------------------------------------------------------------------+
```

Now you can use the TPE in experiment configuration file:

```yaml
tuner:
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
```


## [cifar10-fast](https://github.com/davidcpage/cifar10-fast)

Demonstration of training a small ResNet on CIFAR10 to 94% test accuracy in 79 seconds as described [in this blog series](https://myrtle.ai/learn/how-to-train-your-resnet-1-baseline/).

<img src="net.svg">

Instructions to reproduce on an `AWS p3.2xlarge` instance:
- setup an instance with AMI: `Deep Learning AMI (Ubuntu) Version 11.0` (`ami-c47c28bc` in `us-west-2`) 
- ssh into the instance: `ssh -i $KEY_PAIR ubuntu@$PUBLIC_IP_ADDRESS -L 8901:localhost:8901`
- on the remote machine
    - `source activate pytorch_p36`
    - `pip install pydot` (optional for network visualisation)
    - `git clone https://github.com/davidcpage/cifar10-fast.git`
    - `jupyter notebook --no-browser --port=8901`
 - open the jupyter notebook url in a browser, open `demo.ipynb` and run all the cells

 In my test, 35 out of 50 runs reached 94% test set accuracy with a median of 94.08%. Runtime for 24 epochs is roughly 79s.

 A second notebook `experiments.ipynb` contains code to reproduce the main results from the [posts](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/).

NB: `demo.ipynb` also works on the latest `Deep Learning AMI (Ubuntu) Version 16.0`, but some examples in `experiments.ipynb` trigger a core dump when using TensorCores in versions after `11.0`.
 
## DAWNBench 
 To reproduce [DAWNBench](https://dawn.cs.stanford.edu/benchmark/index.html#cifar10-train-time) timings, setup the `AWS p3.2xlarge` instance as above but instead of launching a jupyter notebook on the remote machine, change directory to `cifar10-fast` and run `python dawn.py` from the command line. Timings in DAWNBench format will be saved to `logs.tsv`. 
 
 Note that DAWNBench timings do not include validation time, as in [this FAQ](https://github.com/stanford-futuredata/dawn-bench-entries), but do include initial preprocessing, as indicated [here](https://groups.google.com/forum/#!topic/dawn-bench-community/YSDRTOLMaMU). DAWNBench timing is roughly 74 seconds which breaks down as 79s (as above) -7s (validation)+ 2s (preprocessing).

## Update 4th Dec 2018
- Core functionality has moved to `core.py` whilst PyTorch specific stuff is in `torch_backend.py` to allow easier experimentation with different frameworks.
- Stats (loss/accuracy) are collected on the GPU and bulk transferred to the CPU at the end of each epoch. This speeds up some experiments so timings in `demo.ipynb` and `experiments.ipynb` no longer match the blog posts.

 


