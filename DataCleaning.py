import numpy as np 
import pandas as pd

import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import csv, collections
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print("Libraries Imported")

df = pd.read_csv('input/training.1600000.processed.noemoticon.csv', encoding='latin', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'tweet'])
df = df.drop(["ids", "date", "flag", "user"], axis=1)
df['target'] = df['target'].replace(4, 1)

## Data Cleaning

# Lower Case
df["tweet"] = df["tweet"].map(lambda x: x.lower())

# Removing Emails
df["tweet"] = df["tweet"].str.replace(r'(\w+\.)*\w+@(\w+\.)+[a-z]+', '')

# Removing URL's
df["tweet"] = df["tweet"].str.replace(r'(http|ftp|https)://[-\w.]+(:\d+)?(/([\w/_.]*)?)?|www[\.]\S+', '')

# Removing hashtag
df["tweet"] = df["tweet"].str.replace(r'[\@\#]\S+', '')

# Converting Emoticons
emo_info = {
    # positive emoticons
    ":‑)": " good ",
    ":)": " good ",
    ";)": " good ",
    ":-}": " good ",
    "=]": " good ",
    "=)": " good ",
    ";d": " good ",
    ":d": " good ",
    ":dd": " good ",
    "xd": " good ",
    ":p": " good ",
    "xp": " good ",
    "<3": " love ",

    # negative emoticons
    ":‑(": " sad ",
    ":‑[": " sad ",
    ":(": " sad ",
    "=(": " sad ",
    "=/": " sad ",
    ":{": " sad ",
    ":/": " sad ",
    ":|": " sad ",
    ":-/": " sad ",
    ":o": " shock "
}

emo_info_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in emo_info.keys()]))]

def emo_repl(phrase):
    for k in emo_info_order:
        phrase = phrase.replace(k, emo_info[k])
    return phrase

df['tweet'] = df['tweet'].apply(emo_repl)

# Expanding Contractions

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\bdon't\b", "do not", phrase)
    phrase = re.sub(r"\bdoesn't\b", "does not", phrase)
    phrase = re.sub(r"\bdidn't\b", "did not", phrase)
    phrase = re.sub(r"\bdidnt\b", "did not", phrase)
    phrase = re.sub(r"\bhasn't\b", "has not", phrase)
    phrase = re.sub(r"\bhaven't\b", "have not", phrase)
    phrase = re.sub(r"\bhavent\b", "have not", phrase)
    phrase = re.sub(r"\bhadn't\b", "had not", phrase)
    phrase = re.sub(r"\bwon't\b", "will not", phrase)
    phrase = re.sub(r"\bwouldn't\b", "would not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)

    # using regular expressions to expand the contractions
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase

df['tweet'] = df['tweet'].apply(decontracted)

# Removing Stop Words

stop = stopwords.words('english')
manual_sw_list = ['retweet', 'retwet', 'rt', 'oh', 'dm', 'mt', 'ht', 'ff', 'shoulda', 'woulda', 'coulda', 'might', 'im', 'tb', 'mysql', 'hah', "a", "an", "the", "and", "but", "if",
                  "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                  "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "nor", "only", "own", "same", "so", "than", "too", "very", "s",
                  "t", "just", "don", "now", 'tweet', 'x', 'f']

stop.extend(manual_sw_list)

df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Applying Lemmatization

lem = WordNetLemmatizer()
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([lem.lemmatize(word, 'v') for word in x.split()]))

# Removing extra punctuations

df["tweet"] = df["tweet"].str.replace(r'[^\w\s]', '')

# Removing Digits

df["tweet"] = df["tweet"].str.replace(r'[0-9]+', '')

# Removing Non-Alphabet

non_alphabet = re.compile(r'[^a-z]+')
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if non_alphabet.search(word) is None]))

# Removing Duplicate Letters

df['tweet'] = df['tweet'].str.replace(r'([a-z])\1{1,}', r'\1\1')
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word if len(wordnet.synsets(word)) > 0 else re.sub(r'([a-z])\1{1,}', r'\1', word) for word in x.split()]))

# Cutting Duplicate Laughing Sound

df['tweet'] = df['tweet'].str.replace(r'(ha)\1{1,}', r'\1')

# Remove Empty Rows

df.drop(df[df["tweet"] == ''].index, inplace=True)
df = df.reset_index(drop=True)

print(df.head())
