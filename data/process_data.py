"""
Preprocessing of Data
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Sample Script Syntax:
> python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db
Arguments Description:
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from two CSV files and merge
    
    Arguments:
        messages_filepath -> Path of the CSV file containing original messages
        categories_filepath -> Path of the CSV file containing different categories
    Output:
        df -> Combined data containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df
    


def clean_data(df):
    """
    Clean data frame
    
    Arguments:
        df -> merged dataframe
    Output:
        df -> cleaned merged dataframe
    """
    # Split categories into separate category columns on the ;
    # create a dataframe named categories of the individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # create a list of category names
    cat_colnames = [category_name.split('-')[0] for category_name in categories.iloc[1]]
    categories.columns = cat_colnames
    
    # encode the category values to 0 or 1 according to the category types
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Replace categories column in df with new category columns.
    # by droping the Categories col and concatenating df and categories data frames.
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], join='inner', axis=1)  
    
    # Drop the duplicates if any
    df.drop_duplicates(inplace=True)
    
    return df



def save_data(df, database_filename):
    """
    Save data to SQLite database
    
    Arguments:
        df -> cleaned merged dataframe
        database_filename -> Path of SQLite database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath)) 
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()