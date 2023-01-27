from imports import *

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
    if expert_type=='naive_fc':
        for e in range(n_experts):
            experts.append(Naive_fc(e, input_dim=input_dim,n_classes=n_classes, latent_dim=2))


    print(f'expert_type: {expert_type}\n'
          f'input_dim: {input_dim}\n'
          f'n_experts: {n_experts}\n'
          f'n_classes: {n_classes}\n'
          f'experts architecture: \n {experts[0]}')
    return experts

