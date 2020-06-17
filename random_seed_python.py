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

train_all = pd.read_csv('train.csv')

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
    
    sampling_results.seed[i] = i
    sampling_results.train_survive[i] = sum(y_train == 1) / X_train.shape[0]
    sampling_results.train_die[i] = sum(y_train == 0) / X_train.shape[0]
    sampling_results.val_survive[i] = sum(y_val == 1) / X_val.shape[0]
    sampling_results.val_die[i] = sum(y_val == 0) / X_val.shape[0]

#######################
# VIEW AND SAVE RESULTS
#######################
    
sampling_results.quantile([0, 1])
    
sampling_results.to_csv('../python_sampling_results_' + str(sampling_rows) + '.csv', index = False)

sr = pd.read_csv('../python_sampling_results_200000.csv')

# Compute change in survival % between training and validation set
sr['survive_diff'] = abs(sr.train_survive - sr.val_survive)

# Plot

fig, axs = plt.subplots(1, 1)
axs.hist(sr.survive_diff, bins = np.arange(0, 0.2, step = 0.01), density = True, edgecolor = 'black', align = 'right')

axs.set_facecolor((242/255, 242/255, 242/255))
plt.xticks(np.arange(0.02, 0.22, step = 0.02))
plt.yticks(np.arange(2, 24, step = 2))
axs.xaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
axs.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 100, decimals = 0))

plt.text(x = 0.11, y = 20, s = "Total Data Splits: 200K", fontsize = 13)
plt.xlabel('Survival % Difference: Training vs. Validation data', fontsize = 13)
plt.ylabel('% of Data Splits', fontsize = 14)
plt.style.use(plt.style.available[11])  # 0, 11, 12, 13, 14, 15, 16
plt.show()

p = sns.distplot(sr.survive_diff, norm_hist = True, bins = 19, kde = False, 
                 hist_kws = dict(edgecolor = "black"), label = "TEST")
p.set(xlabel = 'Survival % Difference between Training and Validation data', 
      ylabel = '% of Data Splits')
#p.set_xticklabels(np.arange(0, 0.2, step = 0.0))
#p.set_yticklabels(np.arange(0, 22, step = 2))
sns.despine()
#p.xticks(np.arange(0, 0.2, step = 0.02))
plt.show()

#####################
# STRATIFIED SAMPLING
#####################

X = X[['Pclass', 'Sex', 'SibSp', 'Fare']]  # These columns are used in the model

# Dummify "Sex" variable
X['gender_dummy'] = pd.get_dummies(X.Sex)['female']
X = X.drop(['Sex'], axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 20200226, stratify = y)

##########
# MODELING
##########

model_rows = 25000

model_results = pd.DataFrame(np.nan, index = range(model_rows), 
                                columns = ['seed', 'acc', 'prec', 'recall'])

for i in range(model_rows):
    
    if i % 1000 == 0: print(i)
    
    # Create and fit model
    clf = RandomForestClassifier(n_estimators = 50, random_state = i)
    clf = clf.fit(X_train, y_train)
    
    preds = clf.predict(X_val)  # Get predictions
    
    model_results.seed[i] = i
    model_results.acc[i] = round(accuracy_score(y_true = y_val, y_pred = preds), 3)
    model_results.prec[i] = round(precision_score(y_true = y_val, y_pred = preds), 3)
    model_results.recall[i] = round(recall_score(y_true = y_val, y_pred = preds), 3)
    
model_results.quantile([0, 1])

model_results.to_csv('../python_model_results_' + str(model_rows) + '.csv', index = False)

mr = pd.read_csv('../python_model_results_25000.csv')

fig, axs = plt.subplots(1, 1)
axs.hist(mr.acc, bins = np.arange(0.76, 0.84, step = 0.01), density = True, edgecolor = 'black', align = 'right')

axs.set_facecolor((242/255, 242/255, 242/255))
#plt.xticks(np.arange(0.02, 0.22, step = 0.02))
#plt.yticks(np.arange(2, 24, step = 2))
axs.xaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
axs.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 100, decimals = 0))

plt.text(x = 0.82, y = 50, s = "Total Models: 25K")
plt.xlabel('Model Accuracy')
plt.ylabel('% of Models')
plt.style.use(plt.style.available[11])  # 0, 11, 12, 13, 14, 15, 16
plt.show()

