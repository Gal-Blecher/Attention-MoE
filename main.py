from config import setup
import torch
import datasets
import build
import train

torch.manual_seed(42)

if __name__ == '__main__':
    dataset = datasets.get_dataset()
    if setup['model_checkpoint_path'] != None:
        model = torch.load(setup['model_checkpoint_path'], map_location=torch.device('cpu'))
    else:
        model = build.build_model()
    if setup['n_experts'] == 1:
        model = model.expert1
        train.train_expert(model, dataset)
    else:
        train.moe_train(model, dataset)
