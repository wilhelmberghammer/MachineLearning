from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils

import matplotlib.pyplot as plt


class VGG_19(nn.Module):
	def __init__(self):
		super(VGG_19, self).__init__()
		# model used: VGG19 (like in the paper)
		# everything after the 28th layer is technically not needed
		self.model = models.vgg19(pretrained=True).features[:30]
		
		# better results when changing the MaxPool layers to AvgPool (-> paper)
		for i, _ in enumerate(self.model):
			# Indicies of the MaxPool layers -> replaced by AvgPool  with same parameters
			if i in [4, 9, 18, 27]:
				self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
		
	def forward(self, x):
		features = []

		for i, layer in enumerate(self.model):
			x = layer(x)
			# indicies of the conv layers after the now AvgPool layers
			if i in [0, 5, 10, 19, 28]:
				features.append(x)
		return features


def load_img(path_to_image, img_size):
	transform = transforms.Compose([
		transforms.Resize((img_size, img_size)),
		transforms.ToTensor(),
	])
	img = Image.open(path_to_image)
	img = transform(img).unsqueeze(0)
	return img


def transfer_style(iterations, optimizer, alpha, beta, generated_image, content_image, style_image, show_images=False):
	for iter in range(iterations+1):
		generated_features = model(generated_image)
		content_features = model(content_image)
		style_features = model(style_image)

		content_loss = 0
		style_loss = 0

		for generated_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
			batch_size, n_feature_maps, height, width = generated_feature.size()

			# in paper it is 1/2*((g - c)**2) ... but it is easies this way because I don't have to worry about dimensions ... and it workes as well
			content_loss += (torch.mean((generated_feature - content_feature) ** 2))

			# batch_size is one ... so it isn't needed. I still inclueded it for better understanding.
			G = torch.mm((generated_feature.view(batch_size*n_feature_maps, height*width)), (generated_feature.view(batch_size*n_feature_maps, height*width)).t())
			A = torch.mm((style_feature.view(batch_size*n_feature_maps, height*width)), (style_feature.view(batch_size*n_feature_maps, height*width)).t())

			# different in paper!!
			E_l = ((G - A)**2)
			# w_l ... one divided by the number of active layers with a non-zero loss-weight -> directly from the paper (technically isn't needed)
			w_l = 1/5
			style_loss += torch.mean(w_l*E_l)

		# I found little difference when changing the alpha and beta values ... still kept it in for better understanding of paper
		total_loss = alpha * content_loss + beta * style_loss
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()


		if iter % 100 == 0:
			print('-'*15)
			print(f'\n{iter} \nTotal Loss: {total_loss.item()} \n Content Loss: {content_loss} \t Style Loss: {style_loss}')
			print('-'*15)

			# show image
			if show_images == True:
				plt.figure(figsize=(10, 10))
				plt.imshow(generated_image.permute(0, 2, 3, 1)[0].cpu().detach().numpy())
				plt.show()
	
	return generated_image

	#if iter % 500 == 0:
		#utils.save_image(generated, f'./gen_{iter}.png')



if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	content_img = load_img('./content_images/alisha.jpg', 512).to(device)
	style_img = load_img('./style_images/dog_abstract_style.jpg', 512).to(device)

	model = VGG_19().to(device)
	# freeze parameters
	for param in model.parameters():
		param.requires_grad = False

	# generated image (init) is the content image ... could also be noise
	# requires_grad because the network itself is frozen ... the thing we are changine is this
	generated_init = content_img.clone().requires_grad_(True)


	iterations = 200
	# the real difference is visibale whe changing the learning rate ... 1e-2 is rather high -> heavy changes to content image
	lr = 1e-2
	# I found no real difference when changing these values...this is why I keep them at 1
	alpha = 1
	beta = 1

	optimizer = optim.Adam([generated_init], lr=lr)

	generated_image = transfer_style(iterations=iterations, 
					optimizer=optimizer, 
					alpha=alpha, 
					beta=beta, 
					generated_image=generated_init, 
					content_image=content_img, 
					style_image=style_img, 
					show_images=False		# only true in jupyter notebook
					)

	utils.save_image(generated_image, f'./gen.png')