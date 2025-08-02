import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()
lemt = WordNetLemmatizer()
StopWords = set(stopwords.words('english'))

def clean_text(raw: str):
    raw = raw.lower()
    raw = BeautifulSoup(raw, "html.parser").get_text(" ")
    raw = re.sub(r'\S+@\S+', '', raw)
    raw = re.sub(r'http\S+|www\S+', '', raw)
    raw = raw.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(raw)
    words = [lemt.lemmatize(w) for w in words if w not in StopWords and w.isalpha()]
    
    
    return ' '.join(words)

