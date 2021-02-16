# Transfer Learning for computer vision

I'll be using the ants and bees dataset like in the pytorch [tutorial on transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)


## Results
```python
batch_size = 8 
EPOCHS = 10

pretrained_model = models.resnet18(pretrained=True)

for param in pretrained_model.parameters():
	param.requires_grad = False

pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 2)
pretrained_model = pretrained_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

model = train(pretrained_model, EPOCHS, criterion, optimizer, data_loaders)
```
```
cpu

__________
EPOCH 0/9
----------
train Loss: 0.6054 Acc: 0.6721
val Loss: 0.4359 Acc: 0.8039
__________
EPOCH 1/9
----------
train Loss: 0.3995 Acc: 0.8648
val Loss: 0.3478 Acc: 0.8431
__________
EPOCH 2/9
----------
train Loss: 0.3758 Acc: 0.8279
val Loss: 0.2956 Acc: 0.9085
__________
EPOCH 3/9
----------
train Loss: 0.2948 Acc: 0.8770
val Loss: 0.2717 Acc: 0.9216
__________
EPOCH 4/9
----------
train Loss: 0.3318 Acc: 0.8607
val Loss: 0.2493 Acc: 0.8954
__________
EPOCH 5/9
----------
train Loss: 0.2259 Acc: 0.8975
val Loss: 0.2445 Acc: 0.8954
__________
EPOCH 6/9
----------
train Loss: 0.2454 Acc: 0.9057
val Loss: 0.2424 Acc: 0.8954
__________
EPOCH 7/9
----------
train Loss: 0.2141 Acc: 0.9303
val Loss: 0.2759 Acc: 0.8627
__________
EPOCH 8/9
----------
train Loss: 0.2456 Acc: 0.8934
val Loss: 0.2233 Acc: 0.9216
__________
EPOCH 9/9
----------
train Loss: 0.1821 Acc: 0.9385
val Loss: 0.2317 Acc: 0.9085
```
