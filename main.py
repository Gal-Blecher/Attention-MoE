from config import setup
import torch
import datasets
import build
import train
import plots
import os
import self_supervised
# from torchsummary import summary

torch.manual_seed(42)

if __name__ == '__main__':
    path = './models/' + setup['experiment_name']
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = datasets.get_dataset()
    if setup['model_eval_path'] != None:
        model = torch.load(setup['model_eval_path'], map_location=torch.device('cpu'))
        if setup['evaluate'] == True:
            dominant_dict = plots.experts_areas(model, dataset['test_loader'])
            points_df = plots.plot_data_latent(model.expert1, dataset['test_loader'])

    else:
        model = build.build_model()
    if setup['n_experts'] == 1:
        model = model.expert1
        # summary(model, (3, 32, 32))
        train.train_expert(model, dataset)
    elif setup['ssl']:
        self_supervised.fit(dataset, model)
    else:
        # summary(model, (3, 32, 32))
        train.moe_train(model, dataset)
