from sklearn.base import BaseEstimator, TransformerMixin


class EmissionsTransformer(BaseEstimator, TransformerMixin):
    """
    Adds dummies to categorical columns and removes the original ones
    """

    def __init__(self, mfs, tmts, tms, fts):
        self.mfs, self.tmts, self.tms, self.fts = mfs, tmts, tms, fts

    def drop_columns(self, X):
        """Dropping irrelevant columns from the data set"""

        irrelevant_numeric = [
            'urban_metric', 'extra_urban_metric', 'urban_imperial',
            'extra_urban_imperial', 'combined_imperial', 'thc_nox_emissions',
            'fuel_cost_6000_miles', 'standard_12_months', 'standard_6_months',
            'first_year_12_months', 'first_year_6_months',
        ]
        X = X.drop(irrelevant_numeric, axis=1)

        irrelevant_categorical = ['model', 'description']
        X = X.drop(irrelevant_categorical, axis=1)

        return X

    def fill_columns(self, X):
        """Filling the numeric columns with the mean of these columns"""

        relevant_numeric = [
            'year', 'euro_standard', 'noise_level', 'engine_capacity',
            'combined_metric', 'fuel_cost_12000_miles', 'co2', 'thc_emissions',
            'co_emissions', 'nox_emissions', 'particulates_emissions',
        ]
        X[relevant_numeric] = X[relevant_numeric].fillna(X.mean())

        return X

    def add_dummies(self, X):
        """Add dummies for every possible value in the data set"""

        for mf in self.mfs:
            X[f'manufacturer_{mf}'] = (X['manufacturer'] == mf).astype(int)
        for tmt in self.tmts:
            X[f'manufacturer_{tmt}'] = (X['manufacturer'] == tmt).astype(int)
        for tm in self.tms:
            X[f'manufacturer_{tm}'] = (X['manufacturer'] == tm).astype(int)
        for ft in self.fts:
            X[f'manufacturer_{ft}'] = (X['manufacturer'] == ft).astype(int)

        X = X.drop(['manufacturer', 'transmission', 'transmission_type', 'fuel_type'], axis=1)
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.drop_columns(X)
        X = self.fill_columns(X)
        X = self.add_dummies(X)
        return X
