import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

def train_ml_model():
    df = pd.read_pickle("data/PreprocessedData/preprocessed_data.pkl")
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(df['processed_review'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #here the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #here the evaluation
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    metrics = {
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr
        }

    #save the model and vectorizer
    with open("models/ml_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    return model, vectorizer

