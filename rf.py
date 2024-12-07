import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from statistics import mean
from tqdm import tqdm
import time


def rfClassify(X_train, y_train,X_test,randseed):
    # creating a RF classifier
    rf = RandomForestClassifier(random_state=randseed)  
    feature_names = [f"{keys[i]}" for i in range(X_test.shape[1])]
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    rf.fit(X_train, y_train)
    # performing predictions on the test dataset
    y_pred = rf.predict(X_test)
    importances = rf.feature_importances_
    std = np.std([importances for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    return y_pred, forest_importances


def randomforestAnalyze(X_train,y_train,X_test,y_test,keys,identifier="rootC",thresholdSig=.01,randseed=1):
   
    y_pred, forest_importances = rfClassify(X_train, y_train,X_test,randseed)
    print(f"---FEATURE IMPORTANCE USING {thresholdSig} THRESHOLD")
    featureShortList = forest_importances.loc[lambda x: x >thresholdSig].sort_values(ascending=False)
    print(featureShortList)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    plt.savefig(f"featImportance_{identifier}.png")
    # metrics are used to find accuracy or error
    from sklearn import metrics      
    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
    #print("Precision OF THE MODEL:", metrics.precision_score(y_test, y_pred))
    #print("Precision OF THE MODEL:", metrics.recall_score(y_test, y_pred))



#this does not work and times out everytime.
def rfe(X,y,randseed=1):

    min_features_to_select = 1  # Minimum number of features to consider
    #rfr = RandomForestRegressor()
    estimator = SVR(kernel="linear")
    # rf = RandomForestClassifier(random_state=randseed)
    cv = StratifiedKFold(5)
    X_norm = MinMaxScaler().fit_transform(X)
    #rfecv = RFECV(estimator=rfr,step=1,cv=cv)
    rfecv = RFECV(estimator, step=1, min_features_to_select = 3)

    rfecv.fit(X_norm, y)

    print(f"Optimal number of features: {rfecv.n_features_}")
    cv_results = pd.DataFrame(rfecv.cv_results_)
    print(cv_results.head())



  