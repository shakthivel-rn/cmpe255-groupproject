import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from itertools import chain
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('Cleaned_Dataset.csv')
df = df[['target', 'tweet']]

X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'], df['target'], test_size=0.25, random_state=0)

print('Visualizing top 5 values')
print('-------------------------')
print(df.head())
print('\n\n')

print('Shape')
print('------')
print(df.shape)
print('\n\n')

vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 3))
vectorizer.fit(X_train)      
x_tr = vectorizer.transform(X_train)
x_te = vectorizer.transform(X_test)

print(f'Vector fitted.')
print('No. of feature_words: ', len(vectorizer.get_feature_names()))
print('\n\n')

def model_Evaluate(model):
    
    # Predict values for Test dataset
    y_pred = model.predict(x_te)

    # Print the evaluation metrics for the dataset.
    print("Performance Report: ")
    print("---------------------")
    print(classification_report(y_test, y_pred))
    print('\n\n')
    
    # Compute and plot the Confusion matrix
    print("Confusion Matrix: ")
    print("---------------------")
    cf_matrix = confusion_matrix(y_test, y_pred)
    print('conf',cf_matrix)


#rf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', max_depth=50,random_state=42)
rf = RandomForestClassifier(n_estimators = 400, min_samples_split=10,min_samples_leaf=1,max_features='sqrt',max_depth= 100, bootstrap= True,random_state = 42)
rf.fit(x_tr, y_train)
model_Evaluate(rf)
