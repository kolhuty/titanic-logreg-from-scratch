import math
import numpy as np
import pandas as pd

class Binary_classifier():

    def __init__(self, X_train, Y_train):
        self.X_train = self.normalize(X_train, 0, 3).astype(float)
        self.Y_train = Y_train.values.astype(float)
        self.weights, self.bias = self.create_random_weights()
        self.sig_z = self.get_probabilities()
        self.n = len(self.sig_z)
        self.d_loss_w, self.d_loss_b = self.gradient()
        self.speed = 0.001
        self.learning()
        self.sig_z = self.get_probabilities()
        self.accuracy = self.accuracy()

    @staticmethod
    def sigmoid(z):
        z = np.array(z, dtype=float)
        return 1/(1+np.exp(-z))

    @staticmethod
    def normalize(X_train: pd.DataFrame, a: int, b: int):
        # нормализация данных
        X_train_scaled = a + (X_train - X_train.min())*(b-a) / (X_train.max() - X_train.min())
        return X_train_scaled.values

    def accuracy(self):
        count = 0
        for i in range(self.n):
            predict = 1 if self.sig_z[i][0] > 0.5 else 0
            if predict == self.Y_train[i]:
                count += 1

        accuracy = count / self.n
        return accuracy

    def create_random_weights(self):
        # веса от -0.1 до 0.1
        np.random.seed(seed=42)
        weights = np.random.uniform(-0.1, 0.1, size=(8, 1))
        bias = np.random.uniform(-0.1, 0.1, size = (1, 1))
        return weights, bias

    def get_probabilities(self):
        # вычисление полученных вероятностей
        z = np.dot(self.X_train, self.weights) + self.bias
        sig_z = self.sigmoid(z)
        return sig_z

    def get_loss(self):
        # вычисление функции потерь
        last_loss = (1/self.n)*(-sum([(self.Y_train[i]*math.log(self.sig_z[i][0]) + (1 - self.Y_train[i])*math.log(1 - self.sig_z[i][0])) for i in range(self.n)]))
        return last_loss

    def gradient(self):
        # градиентный спуск
        d_loss_w = (1 / self.n) * np.dot(self.X_train.T, (self.sig_z - self.Y_train.reshape(-1, 1)))
        d_loss_b = (1 / self.n) * np.sum(self.sig_z - self.Y_train.reshape(-1, 1))
        return d_loss_w, d_loss_b

    def learning(self):
        # градиентный спуск, ищем локальный минимум
        while True:
            self.sig_z = self.get_probabilities()
            self.d_loss_w, self.d_loss_b = self.gradient()
            # вычисление функции потерь
            last_loss = self.get_loss()
            #print(last_loss)
            if last_loss < 0.5: break
            # обновляем веса
            self.weights -= self.speed * self.d_loss_w
            self.bias -= self.speed * self.d_loss_b

# accuracy before learning: 0.5941011235955056
# last loss: 0.49999829353080993
# accuracy after learning: 0.7710674157303371