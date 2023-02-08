import Attention
import data_loading
import MixtureOfExperts
import Training
import torch
import plots
import Experts
from yaml.loader import SafeLoader
import yaml
import nets
import Metrics

def save_vectors(experiment_name, train_acc, test_acc):
    torch.save(train_acc, f'./plots_data/train_acc{experiment_name}.pkl')
    torch.save(test_acc, f'./plots_data/test_acc{experiment_name}.pkl')

def main(n_epochs,
        experts_coeff,
        dataset_name,
        n_experts,
        expert_type,
        experiment_name,
        load_model
        ):
    with open('configuration.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    train_loader, test_loader = data_loading.prepare_data(batch_size=config['dataset'][dataset_name]['batch_size'],
                                                          dataset_name=dataset_name)
    experts = Experts.create_experts(n_experts,
                                     expert_type,
                                     config['dataset'][dataset_name]['n_classes'],
                                     config['dataset'][dataset_name]['input_dim'])

    instance = next(iter(train_loader))[0]
    test = experts[0](instance)
    if n_experts==1:
        model = Experts.ResNet18()
        if load_model is not None:
            model = torch.load(load_model, map_location=torch.device('cpu'))
            acc_l = Metrics.calc_acc(test_loader, model)
            print(f'Loaded model accuracy is: {acc_l}')
        else:
            model, train_loss, train_acc, test_loss, test_acc= \
            Training.full_model_train(train_loader, test_loader, model, n_epochs, experiment_name)
            save_vectors(experiment_name, train_acc, test_acc)
            return model, train_loss, train_acc, test_loss, test_acc
    else:
        router = Attention.AdditiveAttention(input_dim=config['nets'][expert_type]['emb_dim'])
        model = MixtureOfExperts.MoE(experts, router)
        print(model)

    if load_model is not None:
        model = torch.load(load_model, map_location=torch.device('cpu'))
        acc_l = Metrics.calc_acc(test_loader, model)
        print(f'Loaded model accuracy is: {acc_l}')
    # test2 = model(instance)
    model, train_loss, train_acc, test_loss, test_acc =\
        Training.moe_train(train_loader, test_loader, model, n_epochs, experiment_name, experts_coeff)
    save_vectors(experiment_name, train_acc, test_acc)
    return model, train_loss, train_acc, test_loss, test_acc

if __name__ == '__main__':
    torch.manual_seed(42)
    with open('configuration.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    model, train_loss, train_acc, test_loss, test_acc =\
        main(n_epochs=200,
           experts_coeff=0.00001,
           dataset_name='cifar10',
           n_experts=2,
           expert_type='resnet18',
           experiment_name='cifar_2_experts',
           load_model='/Users/galblecher/Desktop/Thesis_out/cifar_2_experts_300_model.pkl'
           )
    


