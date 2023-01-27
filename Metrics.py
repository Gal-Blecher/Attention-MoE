from imports import *
import Training

def metrics(model, data, epoch, train_loss, train_acc, test_loss, test_acc, experiment_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model.eval()
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    for key in data.keys():
        running_loss = 0
        n_correct = 0
        n_tot = 0
        if key == 'train':
            continue
        for i, (images, labels) in enumerate(data[key], start=1):
            images = images.to(device)
            labels = labels.to(device)
            z, out = model(images)
            loss = criterion(out, labels).detach()
            running_loss += loss
            logits = F.softmax(out, dim=0)
            _, predicted = torch.max(logits, 1)
            n_correct += (predicted == labels).sum().item()
            n_tot += labels.shape[0]
            # print('out: ', out)
            # print('logits: ', logits)
            # print('predictions: ', predicted)
            # print('labels: ', labels)
            # break

        loss = running_loss / n_tot
        acc = 100.0 * n_correct / n_tot

        print(f'epoch: {epoch}, {key} loss: {loss}')
        print(f'epoch: {epoch}, {key} acc: {acc}')
        if key == 'train':
            train_loss.append(loss)
            train_acc.append(acc)
        if key == 'test':
            test_loss.append(loss)
            test_acc.append(acc)
            if acc == max(test_acc):
                print('-----------------saving model-----------------')
                torch.save(model, f'./models/{experiment_name}_model.pkl')
                with open(f'./models/{experiment_name}.txt', 'w') as f:
                    f.write(f'n_experts: 1, accuracy: {acc}')

def metrics_moe(model, data, epoch, train_loss, train_acc, test_loss, test_acc, experiment_name, experts_coeff):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model.eval()
    torch.cuda.empty_cache()
    for key in data.keys():
        if key == 'train':
            continue
        running_loss = 0
        n_correct = 0
        n_tot = 0
        for i, (images, labels) in enumerate(data[key], start=1):
            images = images.to(device)
            labels = labels.to(device)
            out, att_weights = model(images)
            logits = F.softmax(out, dim=0)
            _, predicted = torch.max(logits, 1)
            n_correct += (predicted == labels).sum().item()
            n_tot += labels.shape[0]

        loss = running_loss / n_tot
        acc = 100.0 * n_correct / n_tot

        print(f'epoch: {epoch}, {key} acc: {acc}')
        if key == 'train':
            train_loss.append(loss)
            train_acc.append(acc)
        if key == 'test':
            test_loss.append(loss)
            test_acc.append(acc)
            if acc == max(test_acc):
                print('-----------------saving model-----------------')
                torch.save(model, f'./models/{experiment_name}_model.pkl')
                with open(f'./models/{experiment_name}.txt', 'w') as f:
                    f.write(f'n_experts: {model.n_experts} \n experts_coeff: {experts_coeff} \n accuracy: {acc}')
def calc_acc(loader, model):
    n_correct = 0
    n_tot = 0
    for i, (images, labels) in enumerate(loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model((images, labels))[0].squeeze(1)
        logits = F.softmax(outputs, dim=1)
        _, predicted = torch.max(logits, 1)
        n_correct += (predicted == labels).sum().item()
        n_tot += labels.shape[0]
    acc = 100.0 * n_correct / n_tot
    print(acc)