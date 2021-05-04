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

bestparam=clf.best_params_['alpha']   #extracting the best hyperparameter
print("The best Alpha=",bestparam)

mul_model = MultinomialNB(alpha=bestparam) #Building a Naive Bayes model with the best alpha
mul_model.fit(x_tr,y_train)               #Training the model

y_train_pred = mul_model.predict_proba(x_tr)[:,1]  #Prediction using the model(log probability of each class)
y_test_pred = mul_model.predict_proba(x_te)[:,1]
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)   
plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.title("AUC PLOTS")             #Plotting train and test AUC 
plt.grid()
plt.show()

trauc=round(auc(train_fpr, train_tpr),3)
teauc=round(auc(test_fpr, test_tpr),3)
print('Train AUC=',trauc)
print('Test AUC=',teauc)

def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]      #finding the best threashold 
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:                                 #building a confusion matrix with the best threashold 
            predictions.append(0)
    return predictions

best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
TRCM=confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t))
TECM=confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t))

def CM(x,y):
    labels = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in x.flatten()]
                    
    labels = [f"{v1}\n{v2}" for v1, v2 in
    zip(labels,group_counts)]
    labels = np.asarray(labels).reshape(2,2)       #Building a design for the confusion matrix
    sns.heatmap(x, annot=labels, fmt='', cmap='BuPu')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(y)
    plt.plot()

CM(TRCM,'Train Confusion Matrix')

CM(TECM,'Test Confusion Matrix')

joblib.dump(mul_model, 'NaiveBayesModel.pkl')