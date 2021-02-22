import torch
import torch.nn as nn

# this is just an implementation of the CNN model in the paper (-> Figure 3: The Architecture)
class YOLO(nn.Module):
	def __init__(self, in_channels, n_anchorboxes, n_classes):
		super(YOLO, self).__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
			nn.LeakyReLU(.1),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
			nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(.1),
		)

		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(1024*7*7, 4096),
			nn.Dropout(.5),
			nn.LeakyReLU(.1),
			nn.Linear(4096, 7*7*(n_anchorboxes*5 + n_classes))
		)
	
	def forward(self, x):
		x = self.conv_layers(x)
		x = self.fc(x)
		return x