import sys
import numpy as np
import joblib
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report,accuracy_score

def load_data(database_filepath):
    '''
    Loads data from the sql database
    Args
        database_filepath:The file path of the database
    Return
        X:The independent variables (message)
        Y:The dependent variables (categories)
        categories:Categories in which the message can be classified into 
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('data',engine)
    X=df.iloc[:,1]
    Y=df.iloc[:,4:]
    categories=Y.columns.tolist()
    return X,Y,categories
    

def tokenize(text):
    '''
    Cleans the message and converts it into tokens
    Args
        text:The text message
    Return
        final_tokens:Word Tokenized text free of punctuation marks and stopwords
    '''
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    temp_tokens = word_tokenize(text)
    stop = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    final_tokens = [lemmatizer.lemmatize(word) for word in temp_tokens if word not in stop]
    return final_tokens


def build_model():
    '''
   Builds the GridSearchCV model from given parameters
    Args
        None
    Return
        cv:ML model with specified parameters to apply Grid Search on
    '''
        
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators':[10, 25],
        'clf__estimator__min_samples_split':[2,4],
        'tfidf__use_idf': ['True','False'],
        'vect__max_df':[0.8,1.0]
    }
    cv = GridSearchCV(pipeline,param_grid=parameters,n_jobs=1,verbose=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model's performance
    Args
        model:Model fitted on data
        X_test:Validation set of independent variables
        Y_test:Validation set of dependent variables
        category_names:Categories in which the message can be classified into
    Return
        None
    '''
    Y_pred=model.predict(X_test)
    acc=[]
    for i,c in enumerate(category_names):
        print(classification_report(Y_test[c], Y_pred[:,i]))
        acc.append(accuracy_score(Y_test[c], Y_pred[:,i]))
    print('Accuracy :',np.mean(acc))


def save_model(model, model_filepath):
    '''
    Evaluates the model's performance
    Args
        model:Model fitted on data
        model_filepath:File path of the location where the model is to be saved
    Return
        None
    '''   
    joblib.dump(model,model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
    
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
