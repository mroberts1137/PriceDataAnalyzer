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


#lambdas = np.array([3, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200])
#lambdas = np.arange(1, 201)
#warmup_period = lambdas.max()
#polynomial_order = 8
#x_max_length = 20

#x_warmup = pd.read_csv('SPY-daily-2021-2023.csv', header=0, usecols=['Date', 'Close'], nrows=warmup_period)
#x_data = pd.read_csv('SPY-daily-2021-2023.csv', header=0, usecols=['Date', 'Close'], skiprows=range(1,warmup_period+1))


class DataTransform:
    def __init__(self, x_data, lambdas, polynomial_order, x_max_length):

        self.rawdata_dir = os.path.join(os.path.join(os.getcwd(), 'load_dataset'), 'Raw_Data')
        self.data_file = os.path.join(self.rawdata_dir, x_data)

        self.lambdas = lambdas
        self.warmup_period = self.lambdas.max()

        self.x_warmup = pd.read_csv(self.data_file, header=0, usecols=['Date', 'Close'], nrows=self.warmup_period)
        self.x = pd.read_csv(self.data_file, header=0, usecols=['Date', 'Close'], skiprows=range(1, self.warmup_period+1))

        self.mu = np.zeros(len(lambdas))
        self.beta = np.zeros(polynomial_order + 1)
        self.sigma = np.zeros(len(lambdas) + 1)
        self.mu_slope = np.zeros(len(lambdas))
        self.mu_list = []
        # self.beta_df = []
        for i in range(1, len(self.x.index)):
            self.x.loc[i, 'Diff'] = self.x.loc[i, 'Close'] - self.x.loc[i - 1, 'Close']

    def updateData(self, data_new):
        self.x = self.x.append(data_new, ignore_index=True)
        if len(self.x.index) > x_max_length:
            self.x.drop(index=0, inplace=True)
            self.x.reset_index(drop=True, inplace=True)
        for i in range(1, len(self.x.index)):
            self.x.loc[i, 'Diff'] = self.x.loc[i, 'Close'] - self.x.loc[i - 1, 'Close']

        return self.x

    def warmupSMA(self, x_data):
        # x_data = np.pad(x_data, (lambdas.max(), 0), 'edge')
        t0 = len(x_data)

        for i, l in enumerate(lambdas):
            data = x_data[t0 - l: t0]
            self.mu[i] = np.mean(data)
        return self.mu

    def updateMu(self, x_new):
        # run BEFORE updataData since this requires oldest data value, x(t0 - lambda.max)
        # delta_mu ~ dmu/dt = (x(t) - x(t-lambda)) / lambda

        for i, l in enumerate(lambdas):
            t0 = len(self.x.index)
            x_old = self.x.loc[t0 - l, 'Close']
            delta_mu = (x_new - x_old) / l
            self.mu[i] = self.mu[i] + delta_mu
            self.mu_slope[i] = delta_mu

        return self.mu

    def updateBeta(self):
        self.beta = np.polyfit(lambdas, self.mu, polynomial_order)
        return self.beta

    def updateSigma(self):
        for i in range(2, len(lambdas) + 1):
            # take the standard deviation of the first i EMAs
            self.sigma[i] = np.std(self.mu[:i])

        return self.sigma

    def step(self, data_new):
        self.updateMu(data_new['Close'])  # update mu before data since mu requires all old data
        self.updateData(data_new)
        self.updateBeta()
        self.updateSigma()

    def append_mu(self, date):
        new_mu = {l: self.mu[i] for i, l in enumerate(lambdas)}
        new_mu['Date'] = date
        self.mu_list.append(new_mu)

    def batchMu(self, data, warmup_data):
        full_data = np.concatenate((warmup_data, data))
        t0 = len(warmup_data)
        mu = np.zeros((len(data), len(lambdas)))

        for i, l in enumerate(lambdas):
            for j in range(len(data)):
                window = full_data[t0 + j + 1 - l: t0 + j + 1]
                mu[j, i] = np.mean(window)
        return mu

    def batchBeta(self, mu):
        betas = np.zeros((mu.shape[0], polynomial_order + 1))
        for i in range(mu.shape[0]):
            betas[i] = np.polyfit(lambdas, mu[i], polynomial_order)
        return betas

    def batchSigma(self, mu):
        return

    def batchSlope(self):
        return

    def x_from_mu(mu, lambdas):
        # Gives the average value of x between all neighboring values of mu_lambda
        # Time-reversed array is easier to calculate with the ordering of lambda and mu_lambda
        # So we flip the output array

        lambdas = np.insert(lambdas, 0, 0)  # pad with first element=0
        mu = np.insert(mu, 0, 0)
        x_bar = np.zeros(lambdas.max())

        for i in range(len(lambdas) - 1):
            x_bar[lambdas[i]:lambdas[i + 1]] = (lambdas[i + 1] * mu[i + 1] - lambdas[i] * mu[i]) / (
                        lambdas[i + 1] - lambdas[i])

        return np.flip(x_bar)