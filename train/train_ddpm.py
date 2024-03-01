from models import ddpm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

training_data_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

epochs = 1