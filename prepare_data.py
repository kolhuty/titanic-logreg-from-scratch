import pandas as pd

def preparing(train_data: pd.DataFrame):
    # подготовка данных
    train_data.drop(['Ticket','Cabin', 'Name'], axis=1, inplace=True)
    train_data = train_data.dropna()
    train_data.loc[:, 'Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    train_data.loc[:, 'Embarked'] = train_data['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
    # разделяем данные
    X_train = train_data.drop('Survived', axis=1)
    Y_train = train_data['Survived']

    return X_train, Y_train
