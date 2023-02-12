import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import itertools
from config import setup, train_config
import os



def train_expert(model, dataset):
    experiment_name = setup['experiment_name']
    if train_config['device'] == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model = model.to(train_config['device'])
    device = train_config['device']
    print(f'training with device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=setup['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setup['n_epochs'])

    train_acc = []
    test_acc = []
    for epoch in range(setup['n_epochs']):
        model.train()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataset['train_loader']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                print(f'batch_idx: {batch_idx}, train accuracy: {round((correct/total), 2)}')

        acc_train = round((correct/total)*100, 2)
        print(f'epoch: {epoch}, train accuracy: {acc_train}')
        acc_test, test_loss = test(dataset['test_loader'], model)
        train_acc.append(acc_train)
        test_acc.append(acc_test)
        print(f'epoch: {epoch}, test accuracy: {round(acc_test, 2)}')
        scheduler.step()
        if acc_test == max(test_acc):
            print('--------------------------------------------saving model--------------------------------------------')
            path = './models/' + experiment_name
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, f'{path}/model.pkl')

def test(test_loader, model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(train_config['device']), targets.to(train_config['device'])
            _, outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = round((correct / total)*100, 2)
        return acc, loss

def moe_train(model, dataset):
    experiment_name = setup['experiment_name']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    print(f'training with device: {device}')
    router_params = model.router.parameters()
    experts_params = [model.expert1.parameters(), model.expert2.parameters()]
    criterion = nn.CrossEntropyLoss()
    optimizer_experts = optim.SGD(itertools.chain(*experts_params), lr=setup['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler_experts = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_experts, T_max=setup['n_epochs'])
    optimizer_router = optim.SGD(router_params, lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    scheduler_router = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_router, T_max=setup['n_epochs'])

    test_loss = []
    test_acc = []
    for epoch in range(setup['n_epochs']):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataset['train_loader']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_experts.zero_grad()
            optimizer_router.zero_grad()
            outputs, att_weights = model(inputs)
            net_loss = criterion(outputs, targets)
            experts_loss_ = experts_loss(targets, att_weights.squeeze(2), model)
            kl_loss = kl_divergence(att_weights.sum(0))
            loss = net_loss + experts_loss_ + setup['kl_coeff'] * kl_loss

            loss.backward()
            optimizer_experts.step()
            optimizer_router.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 50 == 0:
                print(f'batch_idx: {batch_idx}, train accuracy: {round((correct/total), 2)}')
            if batch_idx % 50 == 0:
                print(f'experts ratio: {att_weights.sum(0).data.T}')
        acc_train = round((correct/total)*100, 2)
        print(f'epoch: {epoch}, train accuracy: {acc_train}')
        acc_test = moe_test(dataset['test_loader'], model)
        test_acc.append(acc_test)
        print(f'epoch: {epoch}, test accuracy: {round(acc_test, 2)}')
        scheduler_experts.step()
        scheduler_router.step()
        if acc_test == max(test_acc):
            print('--------------------------------------------saving model--------------------------------------------')
            path = './models/' + experiment_name
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model, f'{path}/model.pkl')

def moe_test(test_loader, model):
    device = train_config['device']
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, att_weights = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = round((correct / total)*100, 2)
        return acc

def experts_loss(labels, att_weights, model):
    device = train_config['device']
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

def kl_divergence(vector):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = vector.size(0)
    uniform = (torch.ones(n) / n).to(device)
    p = vector / vector.sum()
    return (p * torch.log(p / uniform)).sum()