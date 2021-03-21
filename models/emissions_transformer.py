import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EmissionsTransformer(BaseEstimator, TransformerMixin):
    """
    Adds dummies to categorical columns and removes the original ones
    """

    def drop_columns(self, X):
        """
        Dropping model and description as they would create too many dummies.
        Dropping fuel_cost_6000_miles as it contains only empty values.
        """

        X = X.drop(['model', 'description', 'fuel_cost_6000_miles'], axis=1)
        return X

    def fill_columns(self, X):
        """Filling the numeric columns with the mean of these columns"""

        numeric_columns = X.select_dtypes(exclude=['object']).columns
        X[numeric_columns] = X[numeric_columns].fillna(X.mean())
        return X

    def adjust_categorical(self, X):
        """
        A few issues would occur when adding dummies to the categorical columns without this step.
        """

        small_mf = ['SsangYong', 'Infiniti', 'Bentley Motors', 'Ferrari', 'Maserati',
                    'Lotus', 'Corvette', 'Rolls-Royce', 'Morgan Motor Company', 'Abarth',
                    'Dacia', 'McLaren', 'Perodua', 'MG Motors UK', 'LTI', 'MG Motors Uk']
        X['manufacturer'] = X['manufacturer'].apply(
            lambda row: 'Other' if row in small_mf else row
        )

        small_tm = ['QA6', 'M7', '5AT', 'SAT5', '4AT', 'AMT5', 'A6-AWD', 'A6x2', 'ASM',
                    'DCT7', 'ET5', 'M6-AWD', 'SAT6', 'M6x2', '7SP. SSG', 'MultiDriv',
                    'MultiDrive', 'A8-AWD', 'Multi5', '5MTx2', 'M5x2', 'A5-AWD', 'Multi6',
                    'S/A6', 'MTA5', 'M8']
        X['transmission'] = X['transmission'].apply(
            lambda row: 'Other' if row in small_tm else row
        )

        small_ft = ['Diesel Electric', 'Petrol / E85 (Flex Fuel)', 'Petrol Electric',
                    'Electricity', 'Electricity/Petrol', 'CNG', 'Electricity/Diesel']
        X['fuel_type'] = X['fuel_type'].apply(lambda row: 'Other' if row in small_ft else row)

        return X

    def add_dummies(self, X):
        mf = pd.get_dummies(X['manufacturer'], prefix='manufacturer')
        tm = pd.get_dummies(X['transmission'], prefix='transmission')
        tmt = pd.get_dummies(X['transmission_type'], prefix='transmission_type')
        ft = pd.get_dummies(X['fuel_type'], prefix='fuel_type')

        X = pd.concat([X, mf, tm, tmt, ft], axis=1)
        X = X.drop(['manufacturer', 'transmission', 'transmission_type', 'fuel_type'], axis=1)
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.drop_columns(X)
        X = self.fill_columns(X)
        X = self.adjust_categorical(X)
        X = self.add_dummies(X)
        return X
