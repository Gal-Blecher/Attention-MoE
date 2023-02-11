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
    'device': device
}

setup = {'n_epochs': 200,
         'lr': 0.01,
         'kl_coeff': 0.1,
         'expert_type': 'resnet18',
         'dataset_name': 'cifar10',
         'n_experts': 2,
         'experiment_name': 'resnet18_2_expert_cifar10',
         'model_checkpoint_path': None}


# with open('configuration.yaml', 'w') as f:
#     yaml.dump(train_config, f, sort_keys=False, default_flow_style=False)

def run_config():
    pass

if __name__ == '__main__':
    save_config()
