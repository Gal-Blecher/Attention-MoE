import yaml
from yaml.loader import SafeLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_config = {
    'dataset': {
        'cub200': {
            'n_classes': 200,
            'input_dim': None,
            'batch_size': 8
        },
        'mnist': {
            'n_classes': 10,
            'input_dim': 784,
            'batch_size': 64
        },
        'cifar10': {
            'n_classes': 10,
            'input_dim': None,
            'batch_size': 64
        }
    },
    'nets': {
        'resnet18': {
            'emb_dim': 512
        },
        'vgg16': {
            'emb_dim': 512
        }
    },
    'device': device.type
}

setup = {'n_epochs': 200,
         'lr': 0.01,
         'router_lr': 0.001,
         'kl_coeff': 0.1,
         'expert_type': 'resnet18',
         'dataset_name': 'cifar10',
         'n_experts': 8,
         'experiment_name': 'resnet18_2_expert_cifar10',
         'model_checkpoint_path': '/Users/galblecher/Desktop/Thesis_out/cifar/vgg16/kl/vgg16_4_expert_cifar10_400/model.pkl',
         'evaluate': True}


# with open('configuration.yaml', 'w') as f:
#     yaml.dump(train_config, f, sort_keys=False, default_flow_style=False)

def run_config():
    pass

if __name__ == '__main__':
    pass
