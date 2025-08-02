import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Constants
MAX_NUM_WORDS = 15000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10

def build_tokenizer(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def build_lstm_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate(y_true, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    
    return {
        'accuracy': accuracy,
        'fpr': fpr,
        'fnr': fnr
    }

def train_lstm_model(use_preprocessed=True):
    """
    use_preprocessed = True if you want to use preprocessed data else False
    so you want to choose before running the script.
    """
    if use_preprocessed:
        df = pd.read_pickle("data/PreprocessedData/preprocessed_data.pkl")
        texts = df["processed_review"]
    else:
        df = pd.read_csv("data/ActualData/IMDB Dataset.csv")
        texts = df["review"]
    
    labels = df["sentiment"].map({"positive": 1, "negative": 0})
    
    print(f"Loaded {len(df)} reviews")

    tokenizer = build_tokenizer(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_model(MAX_NUM_WORDS)
    model.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    #evaluation
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    metrics = evaluate(y_test, y_pred)

    #save the model and tokenizer
    suffix = "preprocessed" if use_preprocessed else "raw"
    model.save(f"models/lstm_model_{suffix}.h5")
    with open(f"models/lstm_tokenizer_{suffix}.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return model, tokenizer, metrics
