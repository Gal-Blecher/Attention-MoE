from imports import *
import Metrics
import torch.optim as optim
import yaml
from yaml.loader import SafeLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn


def full_model_train(train_loader, test_loader, model, n_epochs, experiment_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    print(f'training with device: {device}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'epoch: {epoch}, train accuracy: {round((correct/total)*100, 2)}')
        test_acc, test_loss = test(test_loader, model)
        print(f'epoch: {epoch}, test accuracy: {round(test_acc*100, 2)}')

        scheduler.step()
    return model, train_loss, train_acc, test_loss, test_acc

def test(test_loader, model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = round((correct / total)*100, 2)
        return acc, loss

def moe_train(train_loader, test_loader, model, n_epochs , experiment_name, experts_coeff):
    with open(f'./models/{experiment_name}_configuration.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    print(config['training'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training with device: {device}')
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    data = {'train': train_loader, 'test': test_loader}
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['training']['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['training']['step_size']*10), gamma=0.1)
    for epoch in range(n_epochs):
        model.train()
        torch.cuda.empty_cache()
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            outputs, att_weights = model(images)
            net_loss = criterion(outputs, labels)
            experts_loss_ = experts_coeff * experts_loss(labels, att_weights.squeeze(2), model)
            loss = net_loss + experts_loss_
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                lr = optimizer.defaults['lr'], scheduler.get_last_lr()
                print(f'epoch: {epoch}, batch: {i}, loss: {round(running_loss/(100*train_loader.batch_size), 6)}'
                      f', lr: {lr}')
                running_loss = 0
        Metrics.metrics_moe(model, data, epoch, train_loss, train_acc, test_loss, test_acc, experiment_name, experts_coeff)
        scheduler.step()
        if early_stop(test_acc):
            return data, model, train_loss, train_acc, test_loss, test_acc
    return data, model, train_loss, train_acc, test_loss, test_acc


def experts_loss(labels, att_weights, model):
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    if model.n_experts == 2:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels)
            )
            , dim=1)

    if model.n_experts == 4:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels),
            criterion(model.expert3.out, labels),
            criterion(model.expert4.out, labels)
            )
            , dim=1)

    if model.n_experts == 8:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels),
            criterion(model.expert3.out, labels),
            criterion(model.expert4.out, labels),
            criterion(model.expert5.out, labels),
            criterion(model.expert6.out, labels),
            criterion(model.expert7.out, labels),
            criterion(model.expert8.out, labels)
            )
            , dim=1)

    if model.n_experts == 16:
        experts_loss_ = torch.stack(
            (
            criterion(model.expert1.out, labels),
            criterion(model.expert2.out, labels),
            criterion(model.expert3.out, labels),
            criterion(model.expert4.out, labels),
            criterion(model.expert5.out, labels),
            criterion(model.expert6.out, labels),
            criterion(model.expert7.out, labels),
            criterion(model.expert8.out, labels),
            criterion(model.expert9.out, labels),
            criterion(model.expert10.out, labels),
            criterion(model.expert11.out, labels),
            criterion(model.expert12.out, labels),
            criterion(model.expert13.out, labels),
            criterion(model.expert14.out, labels),
            criterion(model.expert15.out, labels),
            criterion(model.expert16.out, labels)
            )
            , dim=1)

    att_weights_flattened = torch.flatten(att_weights)
    experts_loss_flattend = torch.flatten(experts_loss_)
    weighted_experts_loss = torch.dot(att_weights_flattened, experts_loss_flattend)
    return weighted_experts_loss / labels.shape[0]

def early_stop(acc_test_list):
    curr_epoch = len(acc_test_list)
    best_epoch = acc_test_list.index(max(acc_test_list))
    if curr_epoch - best_epoch > 50:
        return True
    else:
        return False
