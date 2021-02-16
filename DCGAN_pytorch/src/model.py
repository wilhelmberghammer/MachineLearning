import torch
import torch.nn as nn



# Discriminator (-> paper)
class Discriminator(nn.Module):
	def __init__(self, img_channels, features_d):
		super(Discriminator, self).__init__()
		self.dis = nn.Sequential(
			
			nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(.2),

			nn.Conv2d(features_d, features_d*2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(features_d*2),
			nn.LeakyReLU(.2),

			nn.Conv2d(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(features_d*4),
			nn.LeakyReLU(.2),

			nn.Conv2d(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(features_d*8),
			nn.LeakyReLU(.2),

			nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
			nn.Sigmoid(),
		)

	def forward(self, x):
		return self.dis(x)



# Generator (-> paper)
class Generator(nn.Module):
	def __init__(self, z_dim, img_channels, features_g):
		super(Generator, self).__init__()
		self.gen = nn.Sequential(

			nn.ConvTranspose2d(z_dim, features_g*16, kernel_size=4, stride=1, padding=0),
			nn.BatchNorm2d(features_g*16),
			nn.ReLU(),

			nn.ConvTranspose2d(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features_g*8),
			nn.ReLU(),

			nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features_g*4),
			nn.ReLU(),

			nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(features_g*2),
			nn.ReLU(),

			nn.ConvTranspose2d(features_g*2, img_channels, kernel_size=4, stride=2, padding=1),
			nn.Tanh(),
		)
	
	def forward(self, x):
		return self.gen(x)



def init_weights(model):
	for i in model.modules():
		if isinstance(i, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
			nn.init.normal_(i.weight.data, .0, .02)

