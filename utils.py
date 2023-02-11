

def save_vectors(experiment_name, train_acc, test_acc):
    torch.save(train_acc, f'./plots_data/train_acc{experiment_name}.pkl')
    torch.save(test_acc, f'./plots_data/test_acc{experiment_name}.pkl')


