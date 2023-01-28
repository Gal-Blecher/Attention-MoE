from imports import *
import nets
import torch.nn.init as init

from torchvision import models

class Resnet18(nn.Module):
    def __init__(self, seed, n_classes):
        super(Resnet18, self).__init__()
        torch.manual_seed(seed=seed)
        resnet18 = models.resnet18(pretrained=True)
        resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        modules = list(resnet18.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.clf = nn.Linear(512, n_classes)
        self.z_dim = 512

    def forward(self, X):
        z = self.model(X).squeeze(2).squeeze(2)
        self.out = self.clf(z)
        logits = F.softmax(self.out, dim=1)
        return z, logits

class Resnet50(nn.Module):
    def __init__(self, seed, n_classes):
        super(Resnet50, self).__init__()
        torch.manual_seed(seed=seed)
        resnet50 = models.resnet50(pretrained=True)
        resnet50.avgpool = nn.AdaptiveAvgPool2d((1,1))
        modules = list(resnet50.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.clf = nn.Linear(2048, n_classes)
        self.z_dim = 2048

    def forward(self, X):
        z = self.model(X).squeeze(2).squeeze(2)
        self.out = self.clf(z)
        logits = F.softmax(self.out, dim=1)
        return z, logits






def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet20(nn.Module):
    def __init__(self, seed, block, num_blocks, n_classes=10):
        super(Resnet20, self).__init__()
        torch.manual_seed(seed)
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, n_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        z = out.view(out.size(0), -1)
        self.out = self.linear(z)
        return z, self.out

class Naive_fc(nn.Module):
    def __init__(self,seed, input_dim, n_classes, latent_dim):
        super(Naive_fc, self).__init__()
        torch.manual_seed(seed=seed)
        self.fc1 = nn.Linear(input_dim, int(input_dim/2))
        self.fc2 = nn.Linear(int(input_dim/2), latent_dim)
        self.fc3 = nn.Linear(latent_dim, n_classes)
        self.z_dim = latent_dim

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        out = self.fc3(z)
        self.out = out
        logits = F.softmax(out, dim=1)
        return z, logits


def create_experts(n_experts, expert_type, n_classes, input_dim):
    experts = []
    if expert_type=='resnet18':
        for e in range(n_experts):
            experts.append(Resnet18(e, n_classes=n_classes))
    if expert_type=='resnet50':
        for e in range(n_experts):
            experts.append(Resnet50(e, n_classes=n_classes))
    if expert_type=='resnet20':
        for e in range(n_experts):
            experts.append(Resnet20(e, BasicBlock, [3, 3, 3], n_classes=n_classes))
    if expert_type=='naive_fc':
        for e in range(n_experts):
            experts.append(Naive_fc(e, input_dim=input_dim,n_classes=n_classes, latent_dim=2))


    print(f'expert_type: {expert_type}\n'
          f'input_dim: {input_dim}\n'
          f'n_experts: {n_experts}\n'
          f'n_classes: {n_classes}\n'
          f'experts architecture: \n {experts[0]}')
    return experts

