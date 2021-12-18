import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class DataProcessing:
    def __init__(self):
        self.data = None
        self.cat_columns = None
        self.transformer = None

    def process(self, data):
        self.data = data
        proc_list = ["_data_preprocess", "_fill_numerical_nan", "_fill_categorical_nan",
                     "_process_rain", "_encoding"]
        for method in proc_list:
            f = getattr(self, method)
            f()
        return self.data

    @staticmethod
    def get_data(path):
        data = pd.read_csv(path)
        return data

    def _data_preprocess(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data.drop('Date', axis=1, inplace=True)
        self.data["RainToday"] = self.data["RainToday"].replace({"Yes": 1, "No": 0})

    def _fill_numerical_nan(self):
        list_of_columns = self.data.select_dtypes(
            include=['float64']).columns  # list of numerical columns to fil, by nan
        list_of_columns = list_of_columns.drop('Rainfall')

        self.data[list_of_columns] = self.data.groupby(["Location", "Month"])[list_of_columns].transform(
            lambda x: x.fillna(x.mean()))
        self.data[list_of_columns] = self.data[list_of_columns].fillna(
            0)  # for cities that all nan fill by 0 TODO add more smart way

        self.data["Rainfall"] = self.data["Rainfall"].fillna(0)

    def _fill_categorical_nan(self):
        self.cat_columns = self.data.select_dtypes(exclude=np.number).columns

        self.data[self.cat_columns] = self.data[self.cat_columns].bfill()

    def _process_rain(self):
        self.data["RainToday"].bfill(inplace=True)

    def process_target(self, y):
        y = y.replace({"Yes": 1, "No": 0})  # FIXME dropna and also in X.
        y = y.fillna(0)

        return y

    def _encoding(self):
        a = pd.get_dummies(self.data.Location, drop_first=True)
        b = pd.get_dummies(self.data.WindGustDir, drop_first=True)
        c = pd.get_dummies(self.data.WindDir9am, drop_first=True)
        d = pd.get_dummies(self.data.WindDir3pm, drop_first=True)
        e = pd.get_dummies(self.data.WindDir3pm, drop_first=True)

        self.data = pd.concat([self.data, a, b, c, d, e], axis=1)
        self.data = self.data.drop(self.cat_columns, axis=1)

    def scaler(self, x_train, x_test):
        self.transformer = RobustScaler()

        x_train = self.transformer.fit_transform(x_train)
        x_test = self.transformer.transform(x_test)

        x_train = pd.DataFrame(x_train, columns=[self.data.columns])
        x_test = pd.DataFrame(x_test, columns=[self.data.columns])

        return x_train, x_test


class DataSplit:
    def __init__(self, test_size=0.3, valid_size=0.1):
        self.test_size = test_size
        self.valid_size = valid_size

    def split(self, data, target='RainTomorrow'):
        y = data[target]
        X = data.drop(target, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=self.valid_size)

        return X_train, X_valid, X_test, y_train, y_valid, y_test
