import optuna
import Main
import yaml
from yaml.loader import SafeLoader



def objective(trial):

    expert_coeff = trial.suggest_float('expert_coeff', 1e-10, 0.1, log=True)
    n_experts = trial.suggest_categorical("n_experts", ['1', '2', '4', '8', '16'])

    with open('configuration.yaml', 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    data, model, train_loss, train_acc, test_loss, test_acc =\
        Main.main(n_epochs=2000,
            experts_coeff=expert_coeff,
            dataset_name='mnist',
            n_experts=int(n_experts),
            expert_type='naive_fc',
            experiment_name='test_naive',
            load_model=None
            )
    return max(test_acc)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)