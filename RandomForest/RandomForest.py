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

# generating word cloud for negative words
from wordcloud import WordCloud

plt.figure(figsize = (15,15)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.target == 0].tweet))
plt.imshow(wc , interpolation = 'bilinear')
#plt.savefig('wordCloudForNagative')

# generating word cloud for positive words
plt.figure(figsize = (15,15)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.target == 1].tweet))
plt.imshow(wc , interpolation = 'bilinear')
#plt.savefig('wordCloudForPositive')

# get a word count per of text
def word_count(words):
    return len(words.split())

# plot word count distribution for both positive and negative 
df['word count'] = df['tweet'].apply(word_count)
p = df['word count'][df.target == 1]
n = df['word count'][df.target == 0]
plt.figure(figsize=(12,6))
plt.xlim(0,45)
plt.xlabel('Word count')
plt.ylabel('Frequency')
g = plt.hist([p, n], color=['g','r'], alpha=0.5, label=['positive','negative'])
plt.legend(loc='upper right')
#plt.savefig('WordCountDistribution')

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
    print('\n\n')

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show()
    #plt.savefig('confusionMatrix')



rf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', max_depth=50,random_state=42)
rf.fit(x_tr, y_train)
model_Evaluate(rf)


joblib.dump(rf, 'RandomForest.pkl')