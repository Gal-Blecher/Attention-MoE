import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

