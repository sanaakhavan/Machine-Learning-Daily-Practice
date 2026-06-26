import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(
    'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)


impute_age = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train[:, 2:3] = impute_age.fit_transform(x_train[:, 2:3])
x_test[:, 2:3] = impute_age.transform(x_test[:, 2:3])

impute_embarked = SimpleImputer(
    missing_values=np.nan, strategy='most_frequent')
x_train[:, 6:7] = impute_embarked.fit_transform(x_train[:, 6:7])
x_test[:, 6:7] = impute_embarked.transform(x_test[:, 6:7])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(
    handle_unknown='ignore'), [1, 6])], remainder='passthrough')
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

sc = StandardScaler()
x_train[:, 5:] = sc.fit_transform(x_train[:, 5:])
x_test[:, 5:] = sc.transform(x_test[:, 5:])
