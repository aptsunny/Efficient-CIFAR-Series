# Experiment

## Speed

Cifar100

### epoch 25
```text
- ConvBnRelu_3 
/ 370s 68？ 

- ConvBnReluPool_3 

/ 440s 76 (worker=32)
/ 380s 72 (worker=16)
/ 340s 72 (worker=8)
/ 340s 72 (worker=4)
/ 380s 74 (worker=0) 

/ 290s 71 (worker=32)(fp16)
/ 210s 75 (worker=8)(fp16)


- ConvBnReluPool_5 810s 71

- ConvBnReluPool+ ConvBnRelu 410s 51 

- ConvBnReluPool_3 + ConvBnReluPool_5 630s 51 

```

### epoch 30
```text
- ConvBnRelu_3 350s 70 (fp16)

```




### DynamicBasicBlock epoch 25
/ 210 67 (worker=8)(fp16)


### DynamicBasicBlock epoch 30
/ 210 67 (worker=8)(fp16) 相同水平