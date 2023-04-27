import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from config import setup
import train

def initial_split(train_loader):
    labeled_indexes = random.sample(range(train_loader.dataset.data.shape[0]), setup['ssl'])
    # Select the remaining 49000 unlabeled examples
    unlabeled_indexes = list(set(range(train_loader.dataset.data.shape[0])) - set(labeled_indexes))
    return labeled_indexes, unlabeled_indexes

def label_samples(model, unlabeled_trainloader, labeled_trainloader, th=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    rm_lst = []
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(unlabeled_trainloader):
            batch_indices = unlabeled_trainloader.sampler.indices[batch_idx * unlabeled_trainloader.batch_size: (batch_idx + 1) * unlabeled_trainloader.batch_size]
            inputs = inputs.to(device)
            outputs = model(inputs)[0]
            outputs = F.softmax(outputs, 1)
            scores, predictions = torch.max(outputs, dim=1)

            # use the inputs, labels, scores, and predictions as needed
            for i in range(inputs.size(0)):
                if scores[i] >= th:
                    labeled_trainloader.sampler.indices.append(batch_indices[i])
                    rm_lst.append(batch_indices[i])
                    labeled_trainloader.dataset.targets[i] = int(predictions[i])
    # unlabeled_trainloader.sampler.indices = [set(unlabeled_trainloader.sampler.indices) - set(rm_lst)]
    unlabeled_trainloader.sampler.indices = list(set(unlabeled_trainloader.sampler.indices) - set(rm_lst))
    model.train()

def fit(dataset, model):
    # logger = get_logger(setup['experiment_name'])
    labeled_indexes, unlabeled_indexes = initial_split(train_loader=dataset['train_loader'])
    orig_labels = dataset['train_loader'].dataset.targets

    labeled_sampler = SubsetRandomSampler(labeled_indexes)
    labeled_trainloader = torch.utils.data.DataLoader(dataset['train_loader'].dataset, batch_size=64,
                                                      sampler=labeled_sampler)
    while True:
        print(f'unlabeled samples: {len(unlabeled_indexes)}')


        unlabeled_sampler = SubsetRandomSampler(unlabeled_indexes)
        unlabeled_trainloader = torch.utils.data.DataLoader(dataset['train_loader'].dataset, batch_size=64, sampler=unlabeled_sampler, shuffle=False)

        dataset_ssl = {
            'train_loader': labeled_trainloader,
            'test_loader': dataset['test_loader']
        }
        train.moe_ssl_train(model, dataset_ssl)

        label_samples(model, unlabeled_trainloader, labeled_trainloader, th=setup['ssl_th'])
        unlabeled_indexes = unlabeled_trainloader.sampler.indices
        dataset['train_loader'] = labeled_trainloader
        if len(unlabeled_indexes) == 0:
            break

    print(f'unlabeled samples: {len(unlabeled_indexes)}')
    print(f'ssl labels: {labeled_trainloader.dataset.targets[:100]}')
    print(f'GT labels: {orig_labels[:100]}')
    return model, labeled_trainloader

