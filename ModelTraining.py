import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from itertools import chain

df = pd.read_csv ('Cleaned_Dataset.csv')
df = df[['target', 'tweet']]
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['target'], test_size = 0.25, random_state = 0)

vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(min_df=10,ngram_range=(1,3)) 
vectorizer.fit(X_train.values)           # Training the TFIDF model
x_tr=vectorizer.transform(X_train.values)
x_te=vectorizer.transform(X_test.values)

model = MultinomialNB()  
parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5,
10, 50, 100]}
clf = GridSearchCV(model, parameters, cv=10,scoring='roc_auc',return_train_score=True)
clf.fit(x_tr, y_train)

results = pd.DataFrame.from_dict(clf.cv_results_)   # converting the results in to a dataframe
results = results.sort_values(['param_alpha'])  
results.head()

train_auc= results['mean_train_score'].values  #extracting the auc scores 
cv_auc = results['mean_test_score'].values

a1=[]
for i in parameters.values():
    a1.append(i)
alphas = list(chain.from_iterable(a1))

plt.plot(alphas, train_auc, label='Train AUC')
plt.plot(alphas, cv_auc, label='CV AUC')
plt.scatter(alphas, train_auc, label='Train AUC points')
plt.scatter(alphas, cv_auc, label='CV AUC points')

plt.legend()
plt.xlabel("Alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("Hyper parameter Vs AUC plot")  
plt.grid()
plt.show()