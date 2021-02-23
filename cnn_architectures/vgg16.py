import torch
import torch.nn as nn



class VGG16(nn.Module):
	'''
		VGG16 (D) architecture acording to paper
	'''
	def __init__(self, input_channels, n_classes):
		super(VGG16, self).__init__()
		self.conv_layers = nn.Sequential(
			# There are way better ways to to this (iterate over a list for example) ... this is so I have a better overview
			nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),

			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),

			nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.fc_layers = nn.Sequential(
			nn.Linear(512*7*7, 4096),	# calculation in paper
			nn.ReLU(),
			nn.Dropout(.5),	# paper: dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5)
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(.5),
			nn.Linear(4096, n_classes),
		)

	
	def forward(self, x):
		x = self.conv_layers(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc_layers(x)
		return x


# the following might differ slightly from the paper
# I didn't do the weight initialisation like in the paper - they trained the shallower model and initialized the larger ones with those weights
model = VGG16(input_channels=3, n_classes=1000)

batch_size = 256
loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters, lr=1e-3, momentum=0.9, weight_decay=5e-4)


# training-loop ...