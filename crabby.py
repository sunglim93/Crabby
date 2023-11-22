import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeCV

class SelectColumns( BaseEstimator, TransformerMixin):
    def __init__(self, columns, featureNum=2):
        self.columns = columns
        self.featureNum = featureNum
        self.selected = columns

    def fit(self, xs, ys, **params):
        ridge = RidgeCV().fit(xs, ys)
        importance = np.abs(ridge.coef_)
        threshold = np.sort(importance)[-(self.featureNum +1)] + 0.01
        sfm = SelectFromModel(ridge, threshold=threshold).fit(xs, ys)
        self.selected = sfm.get_support()
        #print("Columns used:")
        #print(self.columns[self.selected])
        return self
    
    def transform(self, xs):
        return xs[:, self.selected]

def main():
    data = pd.read_csv('train.csv')
    data = pd.get_dummies(data)
    xs = data.drop(columns=['Age'])
    ys = data['Age']

    steps = [
        ('impute', SimpleImputer(strategy='mean')),
        ('column_select', SelectColumns(columns=xs.columns.values)),
    ]
    pipe = Pipeline(steps)
    #add gridsearchcv here

if __name__ == "__main__":
    main()