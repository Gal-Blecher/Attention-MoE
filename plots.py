import torch
import pandas as pd
from sklearn.manifold import TSNE
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datasets


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
                        s=40, legend='full').set(title='1 Expert Latent Representation')
        plt.show()

    return points_df

def experts_areas(model, loader):
  model.eval()
  device = 'cpu'
  # model.to(device)
  dominant_dict = {}
  for i, (images, labels) in enumerate(loader, start=1):
      images = images.to(device)
      labels = labels.to(device)
      # outputs = model(images)[0].squeeze(1)
      weights = model(images)[1].squeeze(1)
      dominant_experts = torch.max(weights, axis=1)[1]
      for l in torch.unique(labels):
        dom_exp, count = torch.unique(dominant_experts[labels==l], return_counts=True)
        for c, exp in enumerate(dom_exp):
          if str(exp.item()) not in dominant_dict.keys():
            dominant_dict[str(exp.item())] = torch.zeros(10).to(device)
          dominant_dict[str(exp.item())][l] += count[c]
  return dominant_dict

# data = datasets.get_dataset('cifar10')
# model_4_expert = torch.load('/Users/galblecher/Desktop/Thesis_out/cifar/vgg16/kl/vgg16_4_expert_cifar10_400/model.pkl', map_location=torch.device('cpu'))
# dominant_dict = experts_areas(model_4_expert, data['test_loader'])