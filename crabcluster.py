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
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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
    xs = data.drop(columns=['Age', 'id'])
    ys = data['Age']

    steps = [
        ('impute', SimpleImputer(strategy='mean')),
        ('column_select', SelectColumns(columns=xs.columns.values)),
        ('scale', MinMaxScaler()),
        ('cluster', KMeans(n_clusters = 4, n_init='auto')),
    ]
    pipe = Pipeline(steps)

    grid = {
    'impute__strategy' : ['mean'],
    'column_select__featureNum' : [7,8,9],
    'column_select__columns' : [xs.columns.values],
    'cluster__n_clusters': range(1,16),
}
    search = GridSearchCV(pipe, grid, scoring='r2')
    search.fit(xs,ys)

    print(search.best_score_)
    print(search.best_params_)

    predictions = search.predict(xs)
    output = data.drop(columns=[col for col in data.columns.values if col != 'id'])
    plotdata = data.drop(columns=[col for col in data.columns.values if col != 'Age'])
    predictions = predictions.round(decimals=1)

    output['yield'] = predictions
    plotdata['Predicted'] = predictions

    output.to_csv('output2.csv', index=False)
    plotdata.to_csv('plotdata2.csv', index=False)

    plt.figure(figsize=(10,10))
    plt.scatter(data['Age'], predictions, c='crimson')
    plt.yscale('linear')
    plt.xscale('linear')

    p1 = max(max(predictions), max(data['Age']))
    p2 = min(min(predictions), min(data['Age']))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Age', fontsize=15)
    plt.ylabel('Predicted', fontsize=15)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()