from torchvision import transforms
import torchvision
import torch
from torchvision.datasets import MNIST
import utils

def prepare_data(batch_size, dataset_name):
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        class_counts = torch.unique(torch.tensor(trainset.targets),return_counts=True)[1]
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=4, shuffle=False)

    if dataset_name == 'mnist':
        train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if dataset_name == 'cub200':
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(448),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])
        transform_test = transforms.Compose([
                transforms.Resize(int(448/0.875)),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])

        train_data = utils.Cub2011('./cub2011', train=True, download=True, transform=transform_train)
        test_data = utils.Cub2011('./cub2011', train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

    print_data_info(train_loader, test_loader, dataset_name)

    return train_loader, test_loader

def print_data_info(train_loader, test_loader, dataset_name):
    if dataset_name == 'cub200':
        train_class_counts = torch.unique(torch.tensor(train_loader.dataset.data.target.values), return_counts=True)[1]
        test_class_counts = torch.unique(torch.tensor(test_loader.dataset.data.target.values), return_counts=True)[1]
    else:
        train_class_counts = torch.unique(torch.tensor(train_loader.dataset.targets), return_counts=True)[1]
        test_class_counts = torch.unique(torch.tensor(test_loader.dataset.targets), return_counts=True)[1]
    print(f'train loader length: {train_loader.dataset.__len__()}')
    print(f'Instances per class in the train loader: \n {train_class_counts}')
    print(f'train loader length: {test_loader.dataset.__len__()}')
    print(f'Instances per class in the test loader: \n {test_class_counts}')
