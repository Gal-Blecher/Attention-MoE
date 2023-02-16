import optuna
from datetime import datetime
from config import setup
import datasets
import train
import build
import os




def objective(trial):
    curr_dt = datetime.now()
    timestamp = int(round(curr_dt.timestamp()))
    setup['experiment_name'] = 'test_naive_2_experts_' + str(timestamp)
    setup['kl_coeff'] = 1 / trial.suggest_int('expert_coeff', 1, 20)
    setup['lr'] = float(trial.suggest_categorical('lr', ['0.1', '0.05', '0.025', '0.01']))
    setup['n_epochs'] = 250
    setup['router_lr'] = setup['lr']
    setup['expert_type'] = 'resnet18'
    setup['dataset_name'] = 'cifar10'
    setup['n_experts'] = 2

    path = './models/' + setup['experiment_name']
    if not os.path.exists(path):
        os.makedirs(path)

    dataset = datasets.get_dataset()
    model = build.build_model()
    train.moe_train(model, dataset)

    return max(model.test_acc)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print(study.best_params)