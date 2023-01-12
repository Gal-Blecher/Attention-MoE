import optuna
import torch

import Main
import yaml
from yaml.loader import SafeLoader
from datetime import datetime




def objective(trial):
    curr_dt = datetime.now()
    timestamp = int(round(curr_dt.timestamp()))
    exp_name = 'test_naive_2_experts_' + str(timestamp)
    expert_coeff = trial.suggest_float('expert_coeff', 1e-10, 0.1, log=True)
    # n_experts = trial.suggest_categorical("n_experts", ['1', '2', '4', '8', '16'])

    with open('configuration.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    data, model, train_loss, train_acc, test_loss, test_acc =\
        Main.main(n_epochs=500,
            experts_coeff=expert_coeff,
            dataset_name='mnist',
            n_experts=2,
            expert_type='naive_fc',
            experiment_name=exp_name,
            load_model=None
            )
    torch.save(train_acc, f'./plots_data/train_acc{exp_name}.pkl')
    torch.save(test_acc, f'./plots_data/test_acc{exp_name}.pkl')
    return max(test_acc)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)