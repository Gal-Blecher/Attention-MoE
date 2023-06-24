import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import itertools
from config import setup, train_config
import os
import json
import pickle
from utils import get_logger
from itertools import cycle


def moe_train_vib(model, dataset):
    logger = get_logger(setup['experiment_name'])
    for key, value in setup.items():
        to_log = str(key) + ': ' + str(value)
        logger.info(to_log)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = './models/' + setup['experiment_name']
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)
    logger.info(f'training with device: {device}')
    router_params = model.router.parameters()
    experts_params = get_experts_params_list(model)
    criterion = nn.CrossEntropyLoss()
    optimizer_experts = optim.SGD(itertools.chain(*experts_params), lr=setup['lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler_experts = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_experts, T_max=setup['n_epochs'])
    optimizer_router = optim.SGD(router_params, lr=setup['router_lr'],
                          momentum=0.9, weight_decay=5e-4)
    scheduler_router = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_router, T_max=setup['n_epochs'])
    model.test_acc = []
    for epoch in range(setup['n_epochs']):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        batch_idx = 1
        if setup['labeled_only'] == True:
            labeled_iter = iter(dataset['labeled_trainloader'])
        else:
            labeled_iter = iter(cycle(dataset['labeled_trainloader']))
        unlabeled_iter = iter(dataset['unlabeled_trainloader'])

        for labeled_data, unlabeled_data in zip(labeled_iter, unlabeled_iter):
            optimizer_experts.zero_grad()
            optimizer_router.zero_grad()

            # labeled data
            labeled_input, targets = labeled_data[0].to(device), labeled_data[1].to(device)
            labeled_output, labeled_att_weights = model(labeled_input)

            labeled_net_loss = setup['classification_loss_coeff_net'] * criterion(labeled_output, targets)
            labeled_experts_loss_ = setup['experts_coeff_labeled'] * experts_loss(targets, labeled_att_weights.squeeze(2), model)
            labeled_kl_loss_router = setup['kl_coeff_router_labeled'] * kl_divergence(labeled_att_weights.sum(0))
            labeled_loss = labeled_net_loss + labeled_experts_loss_ + labeled_kl_loss_router

            # Unlabeled data
            if setup['labeled_only'] == False:
                unlabeled_input = unlabeled_data[0].to(device)
                unlabeled_output, unlabeled_att_weights = model(unlabeled_input)

                unlabeled_experts_loss_ = setup['experts_coeff_unlabeled'] * unlabeled_experts_loss(unlabeled_att_weights.squeeze(2), model)
                unlabeled_kl_loss_router = (setup['kl_coeff_router_unlabeled']) * kl_divergence(unlabeled_att_weights.sum(0))
                unlabeled_loss = unlabeled_experts_loss_ + unlabeled_kl_loss_router
            else:
                unlabeled_loss = 0

            # Loss caculation
            loss = setup['labeled_coeff'] * labeled_loss + setup['unlabeled_coeff'] * unlabeled_loss
            loss.backward()
            optimizer_experts.step()
            optimizer_router.step()

            running_loss += loss.item()
            _, predicted = labeled_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 50 == 0:
                logger.info(f'batch_idx: {batch_idx}, balance between experts: {labeled_att_weights.sum(0).T.detach()}, loss: {round(running_loss/50, 4)}')
                running_loss = 0
            batch_idx += 1

        acc_train = round((correct/total)*100, 2)
        logger.info(f'epoch: {epoch}, train accuracy: {acc_train}')
        scheduler_experts.step()
        scheduler_router.step()

        # Evaluation and model save
        acc_test = moe_test(dataset['testloader'], model)
        model.test_acc.append(acc_test)
        logger.info(f'epoch: {epoch}, test accuracy: {round(acc_test, 2)}')
        if acc_test == max(model.test_acc):
            logger.info('--------------------------------------------saving model--------------------------------------------')
            torch.save(model, f'{path}/model.pkl')
            with open(f"{path}/config.txt", "w") as file:
                file.write(json.dumps(setup))
                file.write(json.dumps(train_config))
            with open(f"{path}/accuracy.txt", "w") as file:
                file.write(f'{epoch}: {acc_test}')
        if early_stop(model.test_acc):
            with open(f'{path}/acc_test.pkl', 'wb') as f:
                pickle.dump(model.test_acc, f)
            return

    with open(f'{path}/acc_test.pkl', 'wb') as f:
        pickle.dump(model.test_acc, f)

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
            setup['criterion_labeled_experts'] * criterion(model.expert1.out, labels) + setup['reconstruction_labeled_experts'] * model.expert1.reconstruction_loss + setup['kl_labeled_experts'] * model.expert1.kl_loss,
            setup['criterion_labeled_experts'] * criterion(model.expert2.out, labels) + setup['reconstruction_labeled_experts'] * model.expert2.reconstruction_loss + setup['kl_labeled_experts'] * model.expert2.kl_loss
            )
            , dim=1)

    att_weights_flattened = torch.flatten(att_weights)
    experts_loss_flattend = torch.flatten(experts_loss_)
    weighted_experts_loss = torch.dot(att_weights_flattened, experts_loss_flattend)
    return weighted_experts_loss / labels.shape[0]

def unlabeled_experts_loss(att_weights, model):
    if model.n_experts == 2:
        experts_loss_ = torch.stack(
            (
            setup['reconstruction_unlabeled_experts'] * model.expert1.reconstruction_loss + setup['kl_unlabeled_experts'] * model.expert1.kl_loss,
            setup['reconstruction_unlabeled_experts'] * model.expert2.reconstruction_loss + setup['kl_unlabeled_experts'] * model.expert2.kl_loss
            )
            , dim=1)


    att_weights_flattened = torch.flatten(att_weights)
    experts_loss_flattend = torch.flatten(experts_loss_)
    weighted_experts_loss = torch.dot(att_weights_flattened, experts_loss_flattend)
    return weighted_experts_loss / att_weights.shape[0]



def kl_divergence(vector):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = vector.size(0)
    uniform = (torch.ones(n) / n).to(device)
    p = vector / vector.sum()
    return (p * torch.log(p / uniform)).sum()

def get_experts_params_list(model):
    if model.n_experts == 2:
        experts_params = [model.expert1.parameters(), model.expert2.parameters()]
        return experts_params
    if model.n_experts == 4:
        experts_params = [model.expert1.parameters(), model.expert2.parameters(),
                          model.expert3.parameters(), model.expert4.parameters()]
        return experts_params
    if model.n_experts == 8:
        experts_params = [model.expert1.parameters(), model.expert2.parameters(),
                          model.expert3.parameters(), model.expert4.parameters(),
                          model.expert5.parameters(), model.expert6.parameters(),
                          model.expert7.parameters(), model.expert8.parameters()]
        return experts_params
    if model.n_experts == 16:
        experts_params = [model.expert1.parameters(), model.expert2.parameters(),
                          model.expert3.parameters(), model.expert4.parameters(),
                          model.expert5.parameters(), model.expert6.parameters(),
                          model.expert7.parameters(), model.expert8.parameters(),
                          model.expert9.parameters(), model.expert10.parameters(),
                          model.expert11.parameters(), model.expert12.parameters(),
                          model.expert13.parameters(), model.expert14.parameters(),
                          model.expert15.parameters(), model.expert16.parameters()]
        return experts_params

def early_stop(acc_test_list):
    curr_epoch = len(acc_test_list)
    best_epoch = acc_test_list.index(max(acc_test_list))
    if curr_epoch - best_epoch > 30:
        return True
    else:
        return False


