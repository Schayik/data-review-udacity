# Udacity Data Review
For the Data Scientist course on Udacity, this is the 'Write a data science blog post' project. I'm using Car Fuel & Emissions data: https://www.kaggle.com/mrmorj/car-fuel-emissions.

## About the data
I was interested in the data, because of the ongoing discussion regarding global warming and the emission of greenhouse gasses. And after taking a closer look at the data, there were a few interesting topics that could be addressed using the data.

- There's tax band labels assigned to the different cars. I'm wondering if it's possible to model these tax band labels using the data.
- Since the years are specified, it would be interesting to see how the environmental impact of cars have evolved during those years.
- It would be interesting to see which companies are ahead of behind in regard on their emissions.


# Project breakdown

- Notebooks
- Flash app
- Data: showing the first look at the raw data and then processing the data to enable modelling.

# Development

## Requirements:
- Python3
- Kaggle account: https://www.kaggle.com

## Libraries:
- pip: to install libraries (included in Python 3)
- venv: to create a virtual environment (included in Python 3)
- Jupyter Notebook: tool that enables us create Jupyter Notebooks (.ipynb files)
- Pandas: enables us to use organised data with labels (uses NumPy)
- sqlalchemy: used to create sql databases
- matplotlib: used to create visualisations by plotting data
- seaborn: create heatmaps
- scikit-learn: library that helps us create machine learning models

## Getting started
1. Clone project from GitHub: `git clone https://github.com/Schayik/u-data-science-blog-post.git`
2. Download data from Kaggle: https://www.kaggle.com/mrmorj/car-fuel-emissions
3. Unzip file and add **data.csv** to the data folder
4. Create virtual environment: `python -m venv .venv`
5. Activate virtual environment: `.venv/Scripts/activate`
6. Install dependencies: `pip install notebook pandas sqlalchemy matplotlib seaborn sklearn`
7. Open Jupyter Notebook: `jupyter notebook`
8. A page in your browser will open: http://localhost:8888/tree
9. Open **data-review.ipynb**

**Note**: I had some troubles with the notebook kernel showing this error: *FileNotFoundError: [WinError 2] The system cannot find the file specified*. I managed to fix it using `python -m ipykernel install --user` from this issue post answer: https://github.com/jupyter/notebook/issues/4079#issuecomment-429651480

## Next time setup
1. Activate virtual environment: `.venv/Scripts/activate`
2. Open Jupyter Notebook: `jupyter notebook`

## Instructions:
1. To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/data.csv data/emissions.db`
2. To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/emissions.db models/classifier.pkl`

**Note**: Run the commands in the project's root directory to set up your database and model.

**Note**: I had memory issues which were resolved by using Python 3.9 64bit instead of Python 3.8 32bit. This article was very helpful: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
