import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import pickle

from load_dataset import DataProcessing

# Hyperparameters:

lambdas = np.array([3, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200])
#lambdas = np.arange(1, 201)
polynomial_order = 8
x_max_length = 20


if __name__ == '__main__':
    dt = DataProcessing.DataTransform('SPY-daily-2021-2023.csv', lambdas, polynomial_order, x_max_length)


