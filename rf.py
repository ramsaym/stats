import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import metrics 
from statistics import mean
from tqdm import tqdm
import time


def rfClassify(X_train, y_train,X_test,keys,randseed):
    # creating a RF classifier
    rf = RandomForestClassifier(random_state=randseed)  
    feature_names = [f"{keys[i]}" for i in range(X_test.shape[1])]
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    rf.fit(X_train, y_train)
    # performing predictions on the test dataset
    y_predictions = rf.predict(X_test)
    importances = rf.feature_importances_
    std = np.std([importances for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    return y_predictions, forest_importances, std


def randomforestAnalyze(X_train,y_train,X_test,y_test,keys,identifier="rootC",thresholdQuant=.25,randseed=1,PLOT=True,METRICS=True):
   
    y_predictions, forest_importances, std = rfClassify(X_train, y_train,X_test,keys,randseed)
    featureCount = forest_importances.shape[0]
    threshhold = forest_importances.quantile(float(thresholdQuant))
    
    
    featureShortList = forest_importances.loc[lambda x: x >float(threshhold)].sort_values(ascending=False)
    print(featureShortList)

    if PLOT:
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
        plt.savefig(f"featImportance_{identifier}.png")
    
    ACCURACY = metrics.accuracy_score(y_test, y_predictions)
    R2 = metrics.r2_score(y_test, y_predictions)
    print(F"ACCURACY OF THE MODEL: {ACCURACY}")

    return featureShortList, ACCURACY, R2



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



def sampleAcrossSeeds(sampling_results,sampling_rows):
    with tqdm(total=sampling_rows) as pbar2:
        for i in range(sampling_rows):
            if i % 10 == 0: pbar2.update(10)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = i)
            sampling_results.loc[i,'seed'] = i
            sampling_results.loc[i,f'{COL}_train'] = sum(y_train) / X_train.shape[0]
            sampling_results.loc[i,f'{COL}_max'] = max(y_train)
            sampling_results.loc[i,f'{COL}_val'] = sum(y_val) / X_val.shape[0]
            sampling_results.loc[i,f'{COL}_max'] = max(y_val)
    sampling_results.to_csv('../_sampling_results_' + str(sampling_rows) + '.csv', index = False)
    sr = pd.read_csv('../_sampling_results_'+ str(sampling_rows) + '.csv')
    print(sr.head())
