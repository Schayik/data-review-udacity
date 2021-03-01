import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from .dummy_transformer import DummyTransformer


def load_data(database_filepath):
    """Loads the data from a database file and returns the target."""

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('emissions', engine)

    X = df.drop(columns='tax_band')
    y = df['tax_band']

    return X, y


def build_model():
    """Describes the model used on the data, consisting of NLP transformers and
    an individual classifier of each category."""

    pipeline = Pipeline([
        ('tfidf', DummyTransformer()),
        ('clf', RandomForestClassifier(n_estimators=10)),
    ])

    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy'],
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
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

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
