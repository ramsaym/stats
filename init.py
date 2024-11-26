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

sampling_rows = 200

sampling_results = pd.DataFrame(np.nan, index = range(sampling_rows), 
                                columns = ['seed', 'train_survive', 'train_die', 'val_survive', 'val_die'])

# Split up dependent and independent variables
X = train_all.drop('Survived', axis = 1)
y = train_all.Survived

for i in range(sampling_rows):
    
    if i % 10 == 0: print(i)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = i,stratify = y)
    #print(X_train.shape[0])
    sampling_results.loc[i,'seed'] = i
    sampling_results.loc[i,'train_survive'] = sum(y_train == 1) / X_train.shape[0]
    sampling_results.loc[i,'train_die'] = sum(y_train == 0) / X_train.shape[0]
    sampling_results.loc[i,'val_survive'] = sum(y_val == 1) / X_val.shape[0]
    sampling_results.loc[i,'val_die'] = sum(y_val == 0) / X_val.shape[0]


#######################
# VIEW AND SAVE RESULTS
#######################


sampling_results.quantile([0, 1])
print(sampling_results.head())    
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

# p = sns.distplot(sr.survive_diff, norm_hist = True, bins = 19, kde = False, 
#                  hist_kws = dict(edgecolor = "black"), label = "TEST")
sns.histplot(data=sr, x="survive_diff",bins=19)

p.set(xlabel = 'Survival % Difference between Training and Validation data', 
      ylabel = '% of Data Splits')
#p.set_xticklabels(np.arange(0, 0.2, step = 0.0))
#p.set_yticklabels(np.arange(0, 22, step = 2))
sns.despine()
#p.xticks(np.arange(0, 0.2, step = 0.02))
plt.show()