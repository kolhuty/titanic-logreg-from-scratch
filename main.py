import pandas as pd
from prepare_data import preparing_train_data, preparing_test_data
from model import Binary_classifier

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
solution = pd.read_csv('data/gender_submission.csv')

X_train, Y_train = preparing_train_data(train_data)
X_test = preparing_test_data(test_data)

bc = Binary_classifier(X_train, Y_train)
print(f"Train accuracy: {bc.train_accuracy}")

test_output = pd.Series(bc.get_predict(X_test))
#print(f"Test output:\n {test_output}")

solution['Survived'] = test_output
solution.to_csv('data/solution.csv', index=False)