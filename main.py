import pandas as pd
from prepare_data import preparing
from model import Binary_classifier

train_data = pd.read_csv('train.csv')
X_train, Y_train = preparing(train_data)

bc = Binary_classifier(X_train, Y_train)
print(bc.accuracy)