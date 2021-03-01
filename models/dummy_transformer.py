from sklearn.base import BaseEstimator, TransformerMixin


class DummyTransformer(BaseEstimator, TransformerMixin):

    def add_dummies(self, X):
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.add_dummies(X)
        return X
