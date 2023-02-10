from imports import *
import MixtureOfExperts
from sklearn.manifold import TSNE

def experts_areas(model, loader):
  model.eval()
  model.to(device)
  dominant_dict = {}
  for i, (images, labels) in enumerate(loader, start=1):
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)[0].squeeze(1)
      weights = model(images)[1].squeeze(1)
      dominant_experts = torch.max(weights, axis=1)[1]
      for l in torch.unique(labels):
        expert_specific_label = dominant_experts[labels==l]
        dom_exp, count = torch.unique(expert_specific_label, return_counts=True)
        for c, exp in enumerate(dom_exp):
          if str(exp.item()) not in dominant_dict.keys():
            dominant_dict[str(exp.item())] = torch.zeros(10).to(device)
          dominant_dict[str(exp.item())][l] += count[c]
  return dominant_dict

def plot_exp_dist(dominant_dict):
  fig, axes = plt.subplots(len(dominant_dict),1, sharex=False, sharey=False, figsize=(10,30))
  for i,key in enumerate(dominant_dict.keys()):
    tot = dominant_dict[key].sum().item()
    print(f'expert {key} is the dominant expert for {tot} instances')
    axes[i].bar(np.arange(10),dominant_dict[key].cpu())
    axes[i].set_title(f'Expert {key}')

def plot_loss(train_loss, test_loss):
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.title('Train/Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend();

def plot_acc(train_acc, test_acc):
    plt.plot(train_acc, label='train')
    plt.plot(test_acc, label='test')
    plt.title('Train/Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend();

def plot_vibs_decision_boundaries(model_vib, data, plot_boundary, show=True):
    z_points = []
    z_labels = []
    model_vib.eval().cpu()
    for d in data['test']:
        z_points.append(model_vib(d)[-1].detach().numpy())
        z_labels.append(d[1].detach().numpy())
    z_points = np.array(z_points).reshape((-1, 2))
    z_labels = np.array(z_labels).flatten()

    z_scores = stats.zscore(z_points)
    abs_zscores = np.abs(z_scores)
    filtered_entries = (abs_zscores < 3).all(axis=1)
    z_points, z_labels = z_points[filtered_entries], z_labels[filtered_entries]

    coords = []
    coords_np = []
    # x_min, x_max = z_points[:, 0].min() - .5, z_points[:, 0].max() + .5
    # y_min, y_max = z_points[:, 1].min() - .5, z_points[:, 1].max() + .5
    x_min, x_max = plot_boundary['x'][0], plot_boundary['x'][1]
    y_min, y_max = plot_boundary['y'][0], plot_boundary['y'][1]
    xx, yy = np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.5)
    for x in xx:
        for y in yy:
            coords.append(torch.tensor([[x, y]]))
            coords_np.append([x, y])
    # coords = np.array(coords)
    coords_np = np.array(coords_np)

    vib_decision = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for c in coords:
        c = c.to(device)
        model_vib = model_vib.to(device)
        logit = model_vib.decoder(c.float())
        pred = torch.max(logit, 1)[1].item()
        vib_decision.append(pred)
    decision_boundary_df = pd.DataFrame({'x': coords_np[:, 0],
                                         'y': coords_np[:, 1],
                                         'label': vib_decision})

    if (show==True):
        sns.scatterplot(data=decision_boundary_df, x='x', y='y', hue='label',
                        legend='full', s=100, palette='Paired').set(title='VIB Decision Boundaries')
        plt.show()

    return decision_boundary_df


def plot_data_latent(model, data, plot_boundary=None, show=True):
    z_points = torch.tensor([])
    z_labels = torch.tensor([])
    model.eval().cpu()
    for d in data:
        z_points = torch.cat((z_points, model(d[0])[0]))
        z_labels = torch.cat((z_labels, d[1]))
    z_points = z_points.detach().numpy()
    z_labels = z_labels.detach().numpy()

    if z_points.shape[1] > 2:
        print('Dim reduction TSNE')
        z_points = TSNE(n_components=2, learning_rate='auto', init='random', random_state=11).fit_transform(z_points)

    points_df = pd.DataFrame({'x': z_points[:, 0],
                              'y': z_points[:, 1],
                              'label': z_labels})

    z_scores = stats.zscore(points_df)
    abs_zscores = np.abs(z_scores)
    filtered_entries = (abs_zscores < 3).all(axis=1)
    # points_df = points_df[filtered_entries]

    # points_df = points_df[(points_df['x'] > plot_boundary['x'][0]) & (points_df['x'] < plot_boundary['x'][1])]
    # points_df = points_df[(points_df['y'] > plot_boundary['y'][0]) & (points_df['y'] < plot_boundary['y'][1])]

    if (show == True):
        sns.scatterplot(data=points_df, x='x', y='y', hue='label', palette='Paired', edgecolor="black",
                        s=40, legend='full').set(title='Expert Latent Representation')
        plt.show()

    return points_df

def plot_summary(model, model_expert, loader, expert_num):
    dominant_dict = experts_areas(model, loader)[str(expert_num-1)].cpu().numpy()
    dominant_df = pd.DataFrame({'x': np.arange(10),
                                'y': dominant_dict})
    plot_boundary = {'x': [-40, 20],
                     'y': [-40, 40]}
    # decision_boundary_df = plot_vibs_decision_boundaries(model_expert, loader, plot_boundary, show=False)
    points_df = plot_data_latent(model_expert, loader, plot_boundary, show=False)
    fig, axes = plt.subplots(1,2, sharex=False, sharey=False, figsize=(30,10))
    # sns.scatterplot(ax=axes[0] , data=decision_boundary_df, x='x', y='y', hue='label',
    #                 legend='full', s=100, palette='Paired').set(title='VIB Decision Boundaries')
    sns.scatterplot(ax=axes[0], data=points_df, x='x', y='y', hue='label', palette='Paired', edgecolor="black",
                        s=40, legend='full').set(title='VIB Latent Representation')
    sns.barplot(ax=axes[1], x='x', y='y', data=dominant_df)
    plt.show()



