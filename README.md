# Sentiment Analysis Project

A comprehensive sentiment analysis system that uses multiple machine learning approaches to classify movie reviews as positive or negative.

## Features

- **Text Preprocessing**: Advanced text cleaning and normalization
- **Logistic Regression Model**: Traditional ML approach using TF-IDF features
- **LSTM Neural Network**: Deep learning model for sequence-based sentiment analysis
- **GPT-2 Classifier**: Transformer-based approach using few-shot learning

## Requirements

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`
- Ypu need to download the dataset into \data\ActualData\ path, before run the main.

## Installation

1. Clone the repository:
```bash
git clone repo url
cd projectroot
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Download NLTK data (required for text preprocessing):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### Complete Pipeline

Run the entire pipeline including preprocessing, training, and evaluation:

```bash
python main.py
```

in the main funtion in main.py, you will see the bellow:
    
    preprocess_the_data()
    train_logistic_regression() 
    train_lstm_model() 
    test_gpt2_classifier()

Each task in the pipeline (preprocessing, logistic regression, LSTM, GPT-2 classification) is controlled by its corresponding function call in `main.py`. To run a specific task, uncomment only the relevant function call and comment out the others, then execute.

e.g:
to run the first task:
if __name__ == "__main__":
    preprocess_the_data()
    # train_logistic_regression() 
    # train_lstm_model() 
    # test_gpt2_classifier()

Each function also includes a test, so you can easily try it out. Before running a test, please remember to comment out the training lines to avoid retraining the model.



## Models

### 1. Logistic Regression
- Uses TF-IDF vectorization
- Fast training and inference

### 2. LSTM Neural Network
- Processes text sequences
- Captures contextual information

### 3. GPT-2 Classifier
- Uses few-shot learning
- No training required
- Good for quick sentiment analysis

## Performance Metrics

Models are evaluated using:
- Accuracy
- False Positive Rate
- False Negative Rate
- Classification Report

## File Descriptions

- `main.py`: Orchestrates the complete pipeline
- `src/gpt2_classifier.py`: GPT-2 based sentiment classification
- `src/lstmModel.py`: LSTM neural network implementation
- `src/trainMLmodel.py`: Traditional ML model training
- `src/utils/preprocessing.py`: Text cleaning and normalization
- `scripts/preprocessdata.py`: Data preprocessing pipeline
- `test_lstm_reviews.py`: Interactive LSTM model testing
- `simple_usage.py`: Basic usage example
