#######
# SETUP
#######

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

train_all = pd.read_csv('titanic_data/train.csv')
print(train_all.head())


################################
# TEST RANDOM SEEDS FOR SAMPLING
################################

sampling_rows = 200000

sampling_results = pd.DataFrame(np.nan, index = range(sampling_rows), 
                                columns = ['seed', 'train_survive', 'train_die', 'val_survive', 'val_die'])

# Split up dependent and independent variables
X = train_all.drop('Survived', axis = 1)
y = train_all.Survived

for i in range(sampling_rows):
    
    if i % 10000 == 0: print(i)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = i)
    #print(X_train.shape[0])
    sampling_results['seed'][i] = i
    sampling_results['train_survive'][i] = sum(y_train == 1) / X_train.shape[0]
    sampling_results['train_die'][i] = sum(y_train == 0) / X_train.shape[0]
    sampling_results['val_survive'][i] = sum(y_val == 1) / X_val.shape[0]
    sampling_results['val_die'][i] = sum(y_val == 0) / X_val.shape[0]

#######################
# VIEW AND SAVE RESULTS
#######################