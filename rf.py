import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from statistics import mean
from tqdm import tqdm


def randomforest(X_train,y_train,X_test,y_test):
    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators = 100)  
    
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)
    
    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    print(clf.feature_importances_)
    
    # metrics are used to find accuracy or error
    from sklearn import metrics  
    print()
    
    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
    #print("Precision OF THE MODEL:", metrics.precision_score(y_test, y_pred))
    #print("Precision OF THE MODEL:", metrics.recall_score(y_test, y_pred))

  