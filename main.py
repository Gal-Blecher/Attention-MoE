from config import setup
import torch
import datasets
import build
import train
import plots
import os

torch.manual_seed(42)

if __name__ == '__main__':
    path = './models/' + setup['experiment_name']
    if not os.path.exists(path):
        os.makedirs(path)
    dataset = datasets.get_dataset()
    if setup['model_eval_path'] != None:
        model = torch.load(setup['model_eval_path'], map_location=torch.device('cpu'))
        if setup['evaluate'] == True:
            dominant_dict = plots.experts_areas(model, dataset['testloader'])
            points_df = plots.plot_data_latent(model.expert1, dataset['testloader'])

    else:
        model = build.build_model()
    if setup['n_experts'] == 1:
        model = model.expert1
        train.train_expert(model, dataset)
    else:
        # if setup['ssl']:
        #     model, dataset = self_supervised.fit(dataset, model)
        train.moe_train_vib(model, dataset)
