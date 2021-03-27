import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from emissions_transformer import EmissionsTransformer


def load_data(database_filepath):
    """Loads the data from a database file and returns the target."""

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('emissions', engine)

    return df


def split_data(df):
    """
    INPUT - df - full car emissions df

    OUTPUT
    X_train, X_test, y_train, y_test - test and train sets
    X_pop - parameter matrix for given tax band labels
    X_nan - parameter matrix of missing tax band values which
        cannot be used for training and testing the model. y_nan would be an empty
    """

    df_pop = df.dropna(subset=['tax_band'])
    X = df_pop.drop(columns='tax_band')
    y = df_pop['tax_band']

    X_train, X_test, y_train, y_test, = train_test_split(X, y)

    df_nan = df[df['tax_band'].isnull()]
    X_nan = df_nan.drop(columns='tax_band')

    return X_train, X_test, y_train, y_test, df_pop, X_nan


def build_model():
    """Describes the model used on the data, consisting of NLP transformers and
    an individual classifier of each category."""

    pipeline = Pipeline([
        ('et', EmissionsTransformer()),
        ('clf', RandomForestClassifier()),
    ])

    parameters = {
        'clf__criterion': ['gini', 'entropy'],
    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, y_test):
    """Shows the accuracy, precision, and recall of the model."""

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(model.best_params_)


def save_model(model, model_filepath):
    """Saves the model as a pickle file"""

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...')
        df = load_data(database_filepath)

        print('Splitting data...')
        X_train, X_test, y_train, y_test, df_pop, X_nan = split_data(df)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
