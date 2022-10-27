import json
import plotly
import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
import re

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


app = Flask(__name__)

def tokenize(text):
    """
    Normalize, tokenize and stems texts.
    
    Input:
        text: Strings, Messages to be tokenized and lemmatized
    
    Output:
        clean_tokens: List of strings, List of tokens extracted from the provided text
    """
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)
        
    # remove punctuation
    # text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    
    # Extract the word tokens from the input text
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words if any
    # words = [w for w in tokens if w not in stopwords.words("english")]
    
    # Lemmatizer to map the words back to its root
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    cleaned_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
    return cleaned_tokens


# Build a custom transformer which extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Tokenize the text function
    
    Arguments:
        text: Strings, Messages to be tokenized and lemmatized
    Output:
        cleaned_tokens: List of strings, List of tokens extracted from the provided text
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP', 'VBD', 'VBG','VBZ', 'VBN'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

    
    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
df = pd.read_sql_table(table_name, engine)

# load model
model = joblib.load("../models/clf.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    labels = df.iloc[:,4:].sum().sort_values(ascending = False).reset_index()
    labels.columns = ['category', 'count']
    label_values = labels['count'].values.tolist()
    label_names = labels['category'].values.tolist()
    
    category_count = df.iloc[:,4:].sum(axis = 0).sort_values(ascending = False)
    category_counts = category_count.head(10)
    category_names = list(category_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=label_names,
                    y=label_values
                )
            ],

            'layout': {
                'title': 'Messages Category Frequency',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Message Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Top Ten Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()