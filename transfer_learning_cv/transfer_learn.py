import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets


def get_data(data_dir, batch_size, num_workers=2):
	data_transforms = {
		'train': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			# because of the resnet model
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]),
		'val': transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]),
	}

	img_datasets = {
		phase: datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase]) for phase in ['train', 'val']
	}

	data_loaders = {
		phase: torch.utils.data.DataLoader(img_datasets[phase], batch_size=batch_size, shuffle=True, num_workers=num_workers) for phase in ['train', 'val']
	}

	class_names = img_datasets['train'].classes

	return data_loaders


def train(model, EPOCHS, criterion, optimizer, data_loaders):
	for epoch in range(EPOCHS):
		print('_' * 10)
		print(f'EPOCH {epoch}/{EPOCHS - 1}')
		print('-' * 10)

		# each epoch has a training and a val phase ... never did that before (from the pytorch tutorial) but makes sense
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = .0
			running_correct = 0

			for x, labels in data_loaders[phase]:
				x = x.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(x)
					loss = criterion(outputs, labels)

					# value is not important (_); index is important (preds)
					_, preds = torch.max(outputs, 1)

					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item()*x.size(0)
				running_correct += torch.sum(preds==labels.data)

			epoch_loss = running_loss / len(data_loaders[phase].dataset)
			epoch_acc = running_correct.double() / len(data_loaders[phase].dataset)

			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

	return model



if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device, '\n')

	# hyperparameters
	batch_size = 8 
	EPOCHS = 10

	data_loaders = get_data(data_dir='./data/', batch_size=batch_size)

	pretrained_model = models.resnet18(pretrained=True)
	for param in pretrained_model.parameters():
		param.requires_grad = False

	pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 2)
	pretrained_model = pretrained_model.to(device)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

	model = train(pretrained_model, EPOCHS, criterion, optimizer, data_loaders)

