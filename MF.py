# -*- coding: utf-8 -*-
# @Time    : 2019-4-16 18:23
# @Author  : Jesse_Michal
# @Email   : 1301873676@qq.com
# @File    : MF.py.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
import matplotlib.pyplot as plt

class MF(object):
    '''
    implement Matrix Factorization(data-P*Q) for Recommend System
    '''

    def __init__(self):
        '''
        To get the train_data and test_data
        Load data "rating.csv", Data format is shown as following, the last column is not very important
        userId,movieId,rating,timestamp
        1,31,2.5,1260759144
        '''
        self.tab = pd.read_csv('ratings.csv')
        self.useri, self.frequsers = np.unique(self.tab.userId, return_counts=True)
        self.itemi, self.freqitems = np.unique(self.tab.movieId, return_counts=True)
        self.n_users = len(self.useri)
        self.n_items = len(self.itemi)

        self.indice_user = pd.DataFrame()
        self.indice_user["indice"] = range(1, len(self.useri) + 1)
        self.indice_user["useri"] = self.useri

        self.indice_item = pd.DataFrame()
        self.indice_item["indice"] = range(1, len(self.itemi) + 1)
        self.indice_item["itemi"] = self.itemi

        x = []
        y = []
        for i in range(0, len(self.tab)):
            x.append((self.indice_user.indice[self.indice_user.useri == self.tab.userId[i]].axes[0] + 1)[0])
            y.append((self.indice_item.indice[self.indice_item.itemi == self.tab.movieId[i]].axes[0] + 1)[0])

        self.tab["userIdnew"] = x
        self.tab["movieIdnew"] = y
        self.train_data, self.test_data = cv.train_test_split(self.tab[["userIdnew", "movieIdnew", "rating"]], test_size=0.25,random_state=123)

    def factorize(self, train_data, test_data, steps, alpha, gamma, k, m, n):
        '''
        To calculate train_error and test_error by matrix factorizing
        :param train_data: the data for train
        :param test_data: the data for test
        :param steps: the epochs
        :param alpha:  learning rate
        :param gamma:  learning rate
        :param k:  number of rows in the matrix Q & P
        :param m:  number of rows in the train_data_matrix
        :param n:  number of columns in the train_data_matrix
        :return:   train_errors and test_errors
        '''
        users, items = train_data.nonzero()
        I = train_data.copy()
        I[I > 0] = 1
        I[I == 0] = 0
        I2 = test_data.copy()
        I2[I2 > 0] = 1
        I2[I2 == 0] = 0
        train_errors = []
        test_errors = []

        P = 3 * np.random.rand(k, m)
        Q = 3 * np.random.rand(k, n)

        for step in range(steps):
            for u, i in zip(users, items):
                e = train_data[u, i] - self.prediction(P[:, u], Q[:, i])
                P[:, u] += gamma * (e * Q[:, i] - alpha * P[:, u])
                Q[:, i] += gamma * (e * P[:, u] - alpha * Q[:, i])

            train_rmse = self.rmse2(I, train_data, Q, P)
            test_rmse = self.rmse2(I2, test_data, Q, P)
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)
        return train_errors, test_errors

    def rmse2(self, I, R, Q, P):
        '''
        To calculate loss
        :param I: 0 or 1 for each element in matrix
        :param R: train_data or test_data
        :return:  loss
        '''
        return np.sqrt(np.sum((I * (R - self.prediction(P, Q))) ** 2) / len(R[R > 0]))

    def prediction(self, P, Q):
        '''
        To predict
        '''
        return np.dot(P.T, Q)

if __name__ == '__main__':

    mf = MF()
    train_data_matrix = np.zeros((mf.n_users, mf.n_items))
    for line in mf.train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
    test_data_matrix = np.zeros((mf.n_users, mf.n_items))
    for line in mf.test_data.itertuples():
        test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    alpha = 0.1
    k = 20
    m, n = train_data_matrix.shape
    steps = 150
    gamma = 0.001

    test_errors, train_errors = mf.factorize(train_data_matrix, test_data_matrix,  steps, alpha, gamma, k, m, n)
    print('RMSE : ' + str(np.mean(test_errors)))

    plt.plot(range(steps), train_errors, marker='o', label='Training Data')
    plt.plot(range(steps), test_errors, marker='v', label='Test Data')
    plt.title('Using SGD')
    plt.xlabel('Number of epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.savefig('./test.jpg')
    plt.show()

