import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(emissions_filepath):
    """Loads the raw data and returns it."""

    df = pd.read_csv(emissions_filepath)
    return df


def clean_data(df):
    """
    INPUT:
    df - (pandas df) emission dataframe to clean

    OUTPUT:
    df -  (pandas df) cleaned emission dataframe
    """

    # Remove unnecessary columns
    df = df.drop(columns=['file', 'date_of_change'])

    # Remove extra spacing
    df['transmission'] = df['transmission'].replace({'ASM  ': 'ASM'})

    return df


def save_data(df, database_filename):
    """Saves the data as a database file"""

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('emissions', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 3:

        emissions_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    EMISSIONS: {emissions_filepath}')
        df = load_data(emissions_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the emissions '
              'dataset as the first argument, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the second argument. \n\nExample: python process_data.py '
              'data.csv '
              'emissions.db')


if __name__ == '__main__':
    main()
