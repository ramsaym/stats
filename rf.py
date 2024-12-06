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
import time


def randomforest(X_train,y_train,X_test,y_test,keys,randseed=0):
    # creating a RF classifier
    rf = RandomForestClassifier(random_state=randseed)  
    feature_names = [f"{keys[i]}" for i in range(X_test.shape[1])]
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    rf.fit(X_train, y_train)
    # performing predictions on the test dataset
    y_pred = rf.predict(X_test)


    start_time = time.time()
    importances = rf.feature_importances_
    std = np.std([importances for tree in rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(importances, index=feature_names)
    print("---FEATURE IMPORTANCE")
    print(forest_importances.loc[lambda x: x >0].sort_values(ascending=False) )

    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # metrics are used to find accuracy or error
    from sklearn import metrics      
    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
    #print("Precision OF THE MODEL:", metrics.precision_score(y_test, y_pred))
    #print("Precision OF THE MODEL:", metrics.recall_score(y_test, y_pred))

  