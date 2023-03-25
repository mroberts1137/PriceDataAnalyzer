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


class DataTransform:
    def __init__(self, x_data, lambdas, polynomial_order, x_max_length, warmup=True):

        self.rawdata_dir = os.path.join(os.path.join(os.getcwd(), 'load_dataset'), 'raw_data')
        self.data_file = os.path.join(self.rawdata_dir, x_data)

        self.warmup = warmup
        self.lambdas = lambdas
        self.warmup_period = self.lambdas.max()
        self.x_max_length = x_max_length
        self.polynomial_order = polynomial_order

        if self.warmup:
            self.x_warmup = pd.read_csv(self.data_file, header=0, usecols=['Date', 'Close'], nrows=self.warmup_period)
            self.x = pd.read_csv(self.data_file, header=0, usecols=['Date', 'Close'], skiprows=range(1, self.warmup_period+1))
        else:
            self.x = pd.read_csv(self.data_file, header=0, usecols=['Date', 'Close'])

        self.mu = np.zeros(len(self.lambdas))
        self.beta = np.zeros(self.polynomial_order + 1)
        self.sigma = np.zeros(len(self.lambdas) + 1)
        self.mu_slope = np.zeros(len(self.lambdas))
        self.mu_list = []
        # self.beta_df = []
        for i in range(1, len(self.x.index)):
            self.x.loc[i, 'Diff'] = self.x.loc[i, 'Close'] - self.x.loc[i - 1, 'Close']
        #self.price_change_mean = self.x['Diff'].mean()
        #self.volatility = self.x['Diff'].std()

    def updateData(self, data_new):
        self.x = self.x.append(data_new, ignore_index=True)
        self.x.iloc[-1]['Diff'] = self.x.iloc[-1]['Close'] - self.x.iloc[-2]['Close']
        if len(self.x.index) > self.x_max_length:
            self.x.drop(index=0, inplace=True)
            self.x.reset_index(drop=True, inplace=True)
        return self.x

    def warmupSMA(self):
        if self.warmup:
            t0 = len(self.x_warmup)
            for i, l in enumerate(self.lambdas):
                self.mu[i] = np.mean(self.x_warmup[t0 - l: t0])
            return self.mu
        else:
            print("Do not run warmupSMA unless warmup=True")

    def updateMu(self, x_new):
        # run BEFORE updataData since this requires oldest data value, x(t0 - lambda.max)
        # delta_mu ~ dmu/dt = (x(t) - x(t-lambda)) / lambda

        t0 = len(self.x.index)
        for i, l in enumerate(self.lambdas):
            x_old = self.x.loc[t0 - l, 'Close']
            delta_mu = (x_new - x_old) / l
            self.mu[i] = self.mu[i] + delta_mu
            self.mu_slope[i] = delta_mu
        return self.mu

    def updateBeta(self):
        self.beta = np.polyfit(self.lambdas, self.mu, self.polynomial_order)
        return self.beta

    def updateSigma(self):
        for i in range(2, len(self.lambdas) + 1):
            # take the standard deviation of the first i EMAs
            self.sigma[i] = np.std(self.mu[:i])
        return self.sigma

    def initiate(self):
        if self.warmup: self.warmupSMA()
        self.updateBeta()
        self.updateSigma()

    def step(self, data_new):
        self.updateMu(data_new['Close'])  # update mu before data since mu requires all old data
        self.updateData(data_new)
        self.updateBeta()
        self.updateSigma()

    def append_mu(self, date):
        new_mu = {l: self.mu[i] for i, l in enumerate(self.lambdas)}
        new_mu['Date'] = date
        self.mu_list.append(new_mu)

    def batchMu(self):
        if self.warmup:
            full_data = np.concatenate((self.x_warmup, self.x))
            t0 = len(self.x_warmup.index)
            mu = np.zeros((len(self.x.index), len(self.lambdas)))

            for i, l in enumerate(self.lambdas):
                for j in range(len(self.x.index)):
                    window = full_data[t0 + j + 1 - l: t0 + j + 1]
                    mu[j, i] = np.mean(window)
            return mu
        else:
            print("Do not run batchMu unless warmup=True")

    def batchBeta(self, mu):
        betas = np.zeros((mu.shape[0], self.polynomial_order + 1))
        for i in range(mu.shape[0]):
            betas[i] = np.polyfit(self.lambdas, mu[i], self.polynomial_order)
        return betas

    def batchSigma(self, mu):
        return

    def batchSlope(self):
        return

    def x_from_mu(self, mu, lambdas):
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