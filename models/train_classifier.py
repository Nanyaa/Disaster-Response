import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql_table('InsertTableName', con=engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)


def tokenize(text):
    """
    Function to enable text tokenization
    
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Train the pipeline
    pipeline.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = pipeline.predict(X_test)
        
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to iterate through columns and call sklearn classification report on each.
    """
    for i, column in enumerate(Y_test):
    return classification_report(Y_test[column], Y_pred[:, i])


def save_model(model, model_filepath):
    import pickle
    model_filepath = 'model.pkl'
    pickle.dump(cv, open(model_filepath, 'wb'))


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