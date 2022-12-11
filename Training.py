from imports import *
import Metrics

def full_model_train(train_loader, test_loader, model, n_epochs, experiment_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training with device: {device}')
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    data = {'train': train_loader, 'test': test_loader}
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(n_epochs):
        model.train()
        torch.cuda.empty_cache()
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            z, out = model(images)
            loss = criterion(out, labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                lr = optimizer.defaults['lr'], scheduler.get_last_lr()
                print(f'epoch: {epoch}, batch: {i}, loss: {round(running_loss/(100*train_loader.batch_size), 6)}'
                      f', lr: {lr}')
                running_loss = 0
        Metrics.metrics(model, data, epoch, train_loss, train_acc, test_loss, test_acc, experiment_name)
        scheduler.step()
    return data, model, train_loss, train_acc, test_loss, test_acc

def moe_train(train_loader, test_loader, model, n_epochs , experiment_name, experts_coeff):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training with device: {device}')
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    data = {'train': train_loader, 'test': test_loader}
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
                print(f'epoch: {epoch}, batch: {i}, loss: {round(running_loss/1000, 4)}')
                running_loss = 0
        Metrics.metrics_moe(model, data, epoch, train_loss, train_acc, test_loss, test_acc, experiment_name)
    return data, model, train_loss, train_acc, test_loss, test_acc


def experts_loss(labels, att_weights, model):
    labels = labels.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    experts_loss_ = torch.stack(
        (
        criterion(model.expert1.out, labels),
        criterion(model.expert1.out, labels)
        )
        , dim=1)
    att_weights_flattened = torch.flatten(att_weights)
    experts_loss_flattend = torch.flatten(experts_loss_)
    weighted_experts_loss = torch.dot(att_weights_flattened, experts_loss_flattend)
    return weighted_experts_loss / labels.shape[0]
