To run MoEAtt, follow these steps:

1. Edit the config.py file with the desired parameters, such as the number of experts, dataset, and expert type.

2. Once the configuration is updated, run the main.py file.

3. After executing main.py, the log file with the model's details will be saved in the "models" folder.

4. We have stored all of the models and running logs in Google Drive. For the sake of anonymous submission, we did not include them, but we can provide them if necessary.

parameters for experimnt - cifar10 rotate resnet18
setup = {'n_epochs': 200,
         'lr': 0.0001,
         'router_lr': 0.0001,
         'kl_coeff': 0.05,
         'experts_coeff': 0.1,
         'expert_type': 'resnet18',
         'dataset_name': 'rotate_cifar10',
         'n_experts': 4,
         'experiment_name': 'cifar10_rotate_resnet_4_experts',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}

parameters for experimnt - cifar10 rotate mobilenet
setup = {'n_epochs': 25,
         'lr': 0.0005,
         'router_lr': 0.0005,
         'kl_coeff': 0.033,
         'experts_coeff': 1.0,
         'expert_type': 'mobilenet',
         'dataset_name': 'rotate_cifar10',
         'n_experts': 4,
         'experiment_name': 'cifar10_rotate_mobilenet_4_experts',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}

parameters for experiment - mnist with naive fc
setup = {'n_epochs': 100,
         'lr': 0.01,
         'router_lr': 0.01,
         'kl_coeff': 0.01,
         'experts_coeff': 1.0,
         'expert_type': 'naive_fc',
         'dataset_name': 'mnist',
         'n_experts': 2/4/8/16,
         'experiment_name': 'mnist_2_expert',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}

parameters for experiment 2 - cifar10 resnet18 2 experts
setup = {'n_epochs': 300,
         'lr': 0.1,
         'router_lr': 0.1,
         'kl_coeff': 0.05,
         'experts_coeff': 0.01,
         'expert_type': 'resnet18',
         'dataset_name': 'cifar10',
         'n_experts': 2,
         'experiment_name': 'cifar_2_expert',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}

parameters for experiment 2 - cifar10 resnet18 4 experts
setup = {'n_epochs': 250,
         'lr': 0.1,
         'router_lr': 0.1,
         'kl_coeff': 0.05,
         'experts_coeff': 0.001,
         'expert_type': 'resnet18',
         'dataset_name': 'cifar10',
         'n_experts': 4,
         'experiment_name': 'cifar_4_expert',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}

parameters for experiment - cifar10 resnet18 8 experts
setup = {'n_epochs': 300,
         'lr': 0.1,
         'router_lr': 0.1,
         'kl_coeff': 0.1,
         'experts_coeff': 0.001,
         'expert_type': 'resnet18',
         'dataset_name': 'cifar10',
         'n_experts': 8,
         'experiment_name': 'cifar_8_expert',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}

parameters for experiment - cifar10 resnet18 16 experts
setup = {'n_epochs': 300,
         'lr': 0.1,
         'router_lr': 0.1,
         'kl_coeff': 0.0555,
         'experts_coeff': 0.1,
         'expert_type': 'resnet18',
         'dataset_name': 'cifar10',
         'n_experts': 16,
         'experiment_name': 'cifar_16_expert',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}

parameters for experiment - cifar10 vgg16 2/4/8/16 experts
setup = {'n_epochs': 300,
         'lr': 0.1,
         'router_lr': 0.1,
         'kl_coeff': 0.05,
         'experts_coeff': 0.01,
         'expert_type': 'vgg16',
         'dataset_name': 'cifar10',
         'n_experts': 2/4/8/16,
         'experiment_name': 'cifar_2-4-8-16_expert',
         'model_checkpoint_path': None,
         'model_eval_path': None,
         'evaluate': False}