import numpy as np
import pandas as pd


class Binary_classifier():

    def __init__(self, X_train: pd.DataFrame, Y_train: pd.DataFrame):
        self.X_train = self.normalize(X_train).astype(float)
        self.Y_train = Y_train.values.astype(float)
        self.weights, self.bias = self.create_random_weights()
        self.sig_z = self.get_probabilities(X_train)
        self.n = len(self.sig_z)
        self.d_loss_w, self.d_loss_b = self.gradient()
        self.speed = 0.001
        self.learning(steps=10000)
        self.train_accuracy = self.accuracy()

    @staticmethod
    def binary_format(x):
        return 1 if x > 0.5 else 0

    @staticmethod
    def sigmoid(z):
        z = np.array(z, dtype=float)
        return 1/(1+np.exp(-z))

    @staticmethod
    def normalize(data: pd.DataFrame, a: float = 0.0, b: float = 1.0):
        # нормализация данных
        data_scaled = a + (data - data.min())*(b-a) / (data.max() - data.min())

        return data_scaled.values

    def accuracy(self):
        count = 0
        for i in range(self.n):
            predict = self.binary_format(self.sig_z[i][0])
            if predict == self.Y_train[i]:
                count += 1

        accuracy = count / self.n
        return accuracy

    def create_random_weights(self):
        # веса от -0.1 до 0.1
        np.random.seed(seed=42)
        weights = np.random.uniform(-0.1, 0.1, size=(self.X_train.shape[1], 1))
        bias = np.random.uniform(-0.1, 0.1, size = (1, 1))
        return weights, bias

    def get_probabilities(self, data):
        # вычисление полученных вероятностей
        z = np.dot(data, self.weights) + self.bias
        sig_z = self.sigmoid(z)
        return sig_z

    def get_loss(self):
        # вычисление функции потерь
        last_loss = (1/self.n)*(
            -np.sum(self.Y_train * np.log(self.sig_z + 1e-9) +
                    (1 - self.Y_train) * np.log(1 - self.sig_z + 1e-9)) )
        return last_loss

    def gradient(self):
        # градиентный спуск
        d_loss_w = (1 / self.n) * np.dot(self.X_train.T, (self.sig_z - self.Y_train.reshape(-1, 1)))
        d_loss_b = (1 / self.n) * np.sum(self.sig_z - self.Y_train.reshape(-1, 1))
        return d_loss_w, d_loss_b

    def learning(self, steps=10000):
        # градиентный спуск
        for _ in range(steps):
            self.sig_z = self.get_probabilities(self.X_train)
            self.d_loss_w, self.d_loss_b = self.gradient()
            self.get_loss()
            # обновляем веса
            self.weights -= self.speed * self.d_loss_w
            self.bias -= self.speed * self.d_loss_b

    def get_predict(self, data: pd.DataFrame) -> np.ndarray:
        data = self.normalize(data, 0, 3).astype(float)
        data_sig_z = self.get_probabilities(data)
        predict = np.apply_along_axis(self.binary_format, axis=1, arr=data_sig_z)

        return predict

# train accuracy before learning: 0.6
# train accuracy after learning: 0.8
# test accuracy: 0.78