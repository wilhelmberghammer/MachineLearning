# Transfer Learning for computer vision

I'll be using the ants and bees dataset like in the pytorch [tutorial on transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)


## Results - Finetuning the conv net
**Result for**
```python
pretrained_model = models.resnet18(pretrained=True), 
			EPOCHS = 8, 
			criterion = nn.CrossEntropyLoss(), 
			optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9), 
			data_loaders)
```
```
cuda 

__________
EPOCH 0/7
----------
train Loss: 0.5661 Acc: 0.6967
val Loss: 0.3534 Acc: 0.8758
__________
EPOCH 1/7
----------
train Loss: 0.3080 Acc: 0.9344
val Loss: 0.2601 Acc: 0.9281
__________
EPOCH 2/7
----------
train Loss: 0.1779 Acc: 0.9549
val Loss: 0.2222 Acc: 0.9281
__________
EPOCH 3/7
----------
train Loss: 0.2191 Acc: 0.9262
val Loss: 0.2124 Acc: 0.9346
__________
EPOCH 4/7
----------
train Loss: 0.1583 Acc: 0.9590
val Loss: 0.2074 Acc: 0.9412
__________
EPOCH 5/7
----------
train Loss: 0.1069 Acc: 0.9754
val Loss: 0.2200 Acc: 0.9346
__________
EPOCH 6/7
----------
train Loss: 0.0986 Acc: 0.9631
val Loss: 0.1850 Acc: 0.9346
__________
EPOCH 7/7
----------
train Loss: 0.0653 Acc: 0.9959
val Loss: 0.1789 Acc: 0.9608
```