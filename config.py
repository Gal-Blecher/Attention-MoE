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
        'rotate_cifar10': {
            'n_classes': 2,
            'input_dim': None,
            'batch_size': 128
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
        },
        'cifar10_ssl': {
            'n_classes': 10,
            'input_dim': None,
            'batch_size': 64
        }
    },
    'nets': {
        'naive_fc': {
            'emb_dim': 2
        },
        'VIB': {
            'emb_dim': 2
        },
        'resnet18': {
            'emb_dim': 512
        },
        'vgg16': {
            'emb_dim': 512
        },
        'resnet50': {
            'emb_dim': 2048
        },
        'basiccnn': {
            'emb_dim': 512
        },
        'mobilenet': {
            'emb_dim': 1280
        },
        'VIBNet': {
            'emb_dim': 128
        },
        'VIBNetResNet': {
            'emb_dim': 512
        }
    },
    'device': device.type
}

setup = {'n_epochs': 100,
         'lr': 0.01,
         'router_lr': 0.01,
         'expert_type': 'VIBNetResNet',
         'dataset_name': 'cifar10_ssl',
         'n_experts': 2,
         'experiment_name': 'VIBNetResNet_500_18_ulo',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False,
         'ssl': 500,
         'labeled_only': False,
         'unlabeled_only': True,
         'unlabeled_coeff': 1.0,
         'labeled_coeff': 0.01,
         'labeled_batch_size': 100,
         'unlabeled_batch_size': 100,
         'experts_coeff_labeled': 0.1,
         'experts_coeff_unlabeled': 0.1,
         'kl_coeff_router_labeled': 0.01,
         'kl_coeff_router_unlabeled': 0.01,
         'classification_loss_coeff_net': 0.1,
         'criterion_labeled_experts': 0.1,
         'reconstruction_labeled_experts': 0,
         'kl_labeled_experts': 0.0000001,
         'reconstruction_unlabeled_experts': 1.0,
         'kl_unlabeled_experts': 0.0000001
         }


# with open('configuration.yaml', 'w') as f:
#     yaml.dump(train_config, f, sort_keys=False, default_flow_style=False)

def run_config():
    pass

if __name__ == '__main__':
    pass
