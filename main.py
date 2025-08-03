import pandas as pd
import pickle
import re
from scripts.preprocessdata import preprocess_data as preprocess_main
from src.trainMLmodel import train_ml_model
from src.utils.preprocessing import clean_text
from sklearn.metrics import accuracy_score, classification_report
from src.lstmModel import train_lstm_model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from src.gpt2_classifier import classify_review
from scripts.preprocessdata import preprocess_data


def preprocess_the_data():
    try:
        preprocess_main()
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

def train_logistic_regression():
    try:
        model, vectorizer = train_ml_model()
        print("Model trained successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        return

    sample_review = "This movie was absolutely fantastic! I loved every minute of it."
    try:
        processed_review = clean_text(sample_review)
        X_review = vectorizer.transform([processed_review])
        prediction = model.predict(X_review)[0]
        confidence = model.predict_proba(X_review)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        conf_score = confidence[1] if prediction == 1 else confidence[0]
        
        print(f"\nReview: {sample_review}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {conf_score:.3f}")
        
    except Exception as e:
        print(f"Error with review: {str(e)}")

def training_lstm_model():
    try:
        use_preprocessed = False  # if you want to use the preproceessed data you want to write True
        model, tokenizer, metrics = train_lstm_model(use_preprocessed=use_preprocessed)  
        print(metrics)
        if use_preprocessed:
            model = load_model('models/lstm_model_preprocessed.h5')
            with open('models/lstm_tokenizer_preprocessed.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
        else:
            model = load_model('models/lstm_model_raw.h5')
            with open('models/lstm_tokenizer_raw.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
        sample_review = "This movie was absolutely fantastic! I loved every minute of it."
        processed_review = clean_text(sample_review)
        X_review = tokenizer.texts_to_sequences([processed_review])
        X_review = pad_sequences(X_review, maxlen=200)
        prediction = model.predict(X_review)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        print(f"\nReview: {sample_review}")
        print(f"Sentiment: {sentiment}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_gpt2_classifier():
    review = "This movie was absolutely fantastic! I loved every minute of it."
    result = classify_review(review, shots=3)
    print(f"Review: {review}")
    print(f"Sentiment: {result}")

if __name__ == "__main__":
    preprocess_the_data()         #uncomment this if you want to preprocess the data
    # train_logistic_regression()       #uncomment this if you want to train the logistic regression model
    # training_lstm_model()        #uncomment this if you want to train the lstm model
    # test_gpt2_classifier()       #uncomment this if you want to test the gpt2 classifier