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
    expert_coeff = trial.suggest_int('expert_coeff', 0, 10)
    lr = trial.suggest_categorical('lr', ['0.1', '0.05', '0.01', '0.005', '0.001'])
    step_size = trial.suggest_int('step_size', 2, 4)
    # n_experts = trial.suggest_categorical("n_experts", ['1', '2', '4', '8', '16'])

    with open('configuration.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)

    config['training']['lr'] = float(lr)
    config['training']['exp_name'] = exp_name
    config['training']['step_size'] = step_size
    config['training']['expert_coeff'] = 10 ** (-expert_coeff)
    with open(f'./models/{exp_name}_configuration.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
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
study.optimize(objective, n_trials=20)
print(study.best_params)