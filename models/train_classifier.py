# Import necessary libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Load data from the specified database file and return X, Y, and category names.
    """
    engine = create_engine('sqlite:///{Disaster_Response.db}')
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenization function to process text data.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define the parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }
    
    # Perform grid search with cross-validation
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print classification reports.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test[category], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1], sys.argv[2]
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
              'as the first argument and the filepath to save the model as '\
              'the second argument. \n\nExample: python train_classifier.py '\
              'InsertDatabaseName.db model.pkl')

if __name__ == '__main__':
    main()