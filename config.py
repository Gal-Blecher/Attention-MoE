import yaml
from yaml.loader import SafeLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_config = {
    'dataset': {
        'cub200': {
            'n_classes': 200,
            'input_dim': None,
            'batch_size': 20
        },
        'cifar100': {
            'n_classes': 100,
            'input_dim': None,
            'batch_size': 128
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
        },
        'resnet50': {
            'emb_dim': 2048
        }
    },
    'device': device.type
}

setup = {'n_epochs': 100,
         'lr': 0.001,
         'router_lr': 0.001,
         'kl_coeff': 0.01,
         'experts_coeff': 1.0,
         'expert_type': 'vgg16',
         'dataset_name': 'cifar100',
         'n_experts': 2,
         'experiment_name': 'test',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}


# with open('configuration.yaml', 'w') as f:
#     yaml.dump(train_config, f, sort_keys=False, default_flow_style=False)

def run_config():
    pass

if __name__ == '__main__':
    pass
