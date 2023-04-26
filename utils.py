import torch
import logging
from config import setup
import torch
import numpy as np
import os
from PIL import Image, TarIO
import pickle
import tarfile

def save_vectors(experiment_name, train_acc, test_acc):
    torch.save(train_acc, f'./plots_data/train_acc{experiment_name}.pkl')
    torch.save(test_acc, f'./plots_data/test_acc{experiment_name}.pkl')


def get_logger(name):
    path = './models/' + setup['experiment_name']
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{path}/log_info.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if handler not in logger.handlers:
        # Add the handler to the logger
        logger.addHandler(handler)

    logger.addHandler(handler)

    return logger


