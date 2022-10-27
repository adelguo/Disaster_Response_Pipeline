# Disaster Response Pipeline Project


### Description:
In this project, we build a model to classify incoming messages received during disasters. By correctly classifying differnet categories of disaster related messages, the correct departments can be assigned and react with as less delay as possible. In this project, basic ETL and machine learning pipelines are built to facilitate the processing of messages. The outputs of the final model will be multilabelled as one message can be related to more than one categories. The data used for model training and testing was downloaded from Figure8.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Required Libraries:
Runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask.

### File Descriptions:
#### File data/process_data.py contains data extraction and cleaning pipeline that:
- Loads the messages and categories dataset
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### File models/train_classifier.py contains machine learning pipeline that:
- Loads data from the SQLite database
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

#### .ipynb - jupyter notebook for preliminary data exploration and Pipeline Preparation

### Licensing, Authors, Acknowledgements:
I am thankful for the resources and guidance provided by Udacity and the trainng dataset provided by Figure8