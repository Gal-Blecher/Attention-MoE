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
        model = experts[0]
        if load_model is not None:
            model = torch.load(load_model)
            plots.plot_data_latent(model, test_loader, plot_boundary=None, show=True)
        else:
            data, model, train_loss, train_acc, test_loss, test_acc= \
            Training.full_model_train(train_loader, test_loader, model, n_epochs, experiment_name)
            save_vectors(experiment_name, train_acc, test_acc)
            return data, model, train_loss, train_acc, test_loss, test_acc
    else:
        router = Attention.AdditiveAttention(input_dim=experts[0].z_dim)
        model = MixtureOfExperts.MoE(experts, router)
        print(model)

    if load_model is not None:
        model = torch.load(load_model)
        plots.plot_summary(model, model.expert1, test_loader, 1)
        return
    # test2 = model(instance)
    data, model, train_loss, train_acc, test_loss, test_acc =\
        Training.moe_train(train_loader, test_loader, model, n_epochs, experiment_name, experts_coeff)
    save_vectors(experiment_name, train_acc, test_acc)
    return data, model, train_loss, train_acc, test_loss, test_acc

if __name__ == '__main__':
    torch.manual_seed(42)
    with open('configuration.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    data, model, train_loss, train_acc, test_loss, test_acc =\
        main(n_epochs=100,
           experts_coeff=0.00001,
           dataset_name='cifar10',
           n_experts=1,
           expert_type='resnet20',
           experiment_name='x',
           load_model=None
           )
    


