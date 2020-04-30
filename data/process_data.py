import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads dataset from the given path
    Args
        messages_filepath:The file path of the messages dataset
        categories_filepath:The file path of the categories dataset
    Return
        df:The dataset after merging both the datasets on common column id
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories,on='id')
    return df

def clean_data(df):
    '''
    Cleans the dataset
    Args
        df:The dataset after merging both the datasets on common column id 
    Return
        df:The final dataset after cleaning
    '''
    categories = df['categories'].str.split(';',expand=True)
    row_names = categories.iloc[0,:]
    category_colnames = row_names.transform(lambda x:x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].transform(lambda x:x[-1:])
        categories[column] = pd.to_numeric(categories[column])
    df.drop(columns=['categories'],axis=1,inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    daves the dataset to a sql database
    Args
        df:The final dataset after cleaning
        database_filename:The file path of the database
    Return
        None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('data', engine, index=False)


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