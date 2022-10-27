import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

import os
import re
from sqlalchemy import create_engine
import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

from nltk.corpus import stopwords

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


def load_data(database_filepath):
    """
    Load the data
    
    Inputs:
        database_filepath: String. Filepath for the db file containing the cleaned data.
    
    Output:
        X: Dataframe. Contains the feature data.
        y: Dataframe. Contains the labels (categories) data.
        category_names: List of strings. Contains the labels names.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    X = df['message']
    y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    category_names = y.columns.tolist()
    
    return X, y, category_names


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

    
    
def build_model():
    """
    Builds a ML pipeline and performs gridsearch.
    Args:
        None
    Returns:
        grid: optimized GridSearchCV object.
    """
    
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())])
        ),
        ('starting_verb', StartingVerbExtractor())])
     ),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    grid = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', error_score='raise')
    
    return grid



def evaluate_model(model, X_test, y_test, category_names):
    """
    Returns test accuracy, number of 1s and 0s, recall, precision and F1 Score.
    
    Inputs:
        model: model object
        X_test: pandas dataframe containing test features.
        y_test: pandas dataframe containing test labels.
        category_names: list of strings containing category names.
    
    Returns:
        None
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test.values, y_pred, target_names = category_names))
    

    
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model.best_estimator_, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()