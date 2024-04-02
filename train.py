import torch
import numpy as np


def trainer(model, optimizer, train_loader,epoch):
    nb_loss = 0
    ct_loss = 0
    sum_loss = 0
    for 