import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model import Discriminator, Generator, init_weights


# cuda if GPU is available, otherwise train on cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} for training!")

# hyperparameters (-> paper)
img_size = 64
img_channels = 1	# 3 for rgb
z_dim = 100			# noise dim
feature_dis = 64
feature_gen = 64

EPOCHS = 1
batch_size = 64
learning_rate = 2e-4


# Transforms
transforms = transforms.Compose([
	transforms.Resize(img_size),
	transforms.ToTensor(),
	transforms.Normalize([.5], [.5]), 	# needs to be changes when using more then 1 channel!!
])

# download and load dataset - in this case I'll be using MNIST
dataset = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transforms)
# create datalaoder
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# init Generator
gen = Generator(z_dim, img_channels, feature_gen).to(device)
dis = Discriminator(img_channels, feature_dis).to(device)

# init weights
init_weights(gen)
init_weights(dis)

# optimizers (-> paper)
gen_optim = optim.Adam(gen.parameters(), lr=learning_rate, betas=(.5, .999))
dis_optim = optim.Adam(dis.parameters(), lr=learning_rate, betas=(.5, .999))

# loss function
loss = nn.BCELoss()


# training loop
print('starting training loop...')
for epoch in range(EPOCHS):
	for batch, (real_img, _) in enumerate(data_loader):
		real_img = real_img.to(device)
		z = torch.randn((batch_size, z_dim, 1, 1)).to(device)	# noise
		fake_img = gen(z)

		# Discriminator training
		dis_real_img = dis(real_img).reshape(-1)
		loss_dis_real_img = loss(dis_real_img, torch.ones_like(dis_real_img))

		dis_fake_img = dis(fake_img).reshape(-1)
		loss_dis_fake_img = loss(dis_fake_img, torch.zeros_like(dis_fake_img))

		loss_dis = (loss_dis_real_img + loss_dis_fake_img)/2
		dis.zero_grad()
		loss_dis.backward(retain_graph = True)		# retain_graph because we are using it in the training for the generator
		dis_optim.step()


		# Generator training
		out = dis(fake_img).reshape(-1)
		loss_gen = loss(out, torch.ones_like(out))
		gen.zero_grad()
		loss_gen.backward()
		gen_optim.step()

	print(f"{epoch}/{EPOCHS}")

	# print a grid after every epoch - tensorboard would be better but that is a pain in colab so...
	with torch.no_grad():
		fake_img = gen(fixed_z)
		img_grid_fake = torchvision.utils.make_grid(fake_img[:32], normalize=True).cpu()
		plt.imshow(img_grid_fake.permute(1, 2, 0))
		plt.show()
