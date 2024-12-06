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
from statistics import mean
from tqdm import tqdm
import sys


#####CONFIG######################################################
#################################################################
name_of_script = sys.argv[0]
datafile = sys.argv[1]
# datacolumn = sys.argv[2]
# analysisType = sys.argv[3]
train_all = pd.read_csv(datafile)
#train_all = pd.read_csv("dndc_data/randomSet_1_dndc.csv")
sampling_rows = 200
sampling_results = pd.DataFrame(np.nan, index = range(sampling_rows), columns = ['seed', 'rootc_train', 'rootc_max', 'rootc_val', 'rootc_max'])
ftrain = train_all[train_all['Crop 1.23_RootC'] > 0]
#print(ftrain.keys())
X = ftrain.drop('Crop 1.23_RootC', axis = 1) # Split up dependent and independent variables
y = ftrain['Crop 1.23_RootC'].astype('int64')
identifier='rootc'
columnsofinterest=['x1','y1','Crop 1.23_RootC','Resistant litter','SOC10-20cm','SOC30-40cm',
                   'SOC50-60cm','Microbe','Humads','Humus','DayPET_Crop(mm)',"Radiation(MJ/m2/d)",'Prec.(mm)','Temp.(C)']
VERBOSE=True
SAMPLE=False
#####CONFIG######################################################
################################################################
if VERBOSE: print(ftrain.loc[:,columnsofinterest].head())

if SAMPLE:
    print(f"SAMPLING FOR SEED SENSTIVITY: {sampling_rows} iterations")
    ####PROCESS#####################################################
    ################################################################
    with tqdm(total=sampling_rows) as pbar2:
        for i in range(sampling_rows):
            if i % 10 == 0: pbar2.update(10)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = i)
            sampling_results.loc[i,'seed'] = i
            sampling_results.loc[i,f'{identifier}_train'] = sum(y_train) / X_train.shape[0]
            sampling_results.loc[i,f'{identifier}_max'] = max(y_train)
            sampling_results.loc[i,f'{identifier}_val'] = sum(y_val) / X_val.shape[0]
            sampling_results.loc[i,f'{identifier}_max'] = max(y_val)

    sampling_results.to_csv('../_sampling_results_' + str(sampling_rows) + '.csv', index = False)
    sr = pd.read_csv('../_sampling_results_'+ str(sampling_rows) + '.csv')
    print(sr.head())


###APP0 - TRAIN ON A FIXED SEED AND CLASSIFY WITH RF
from rf import randomforestClassify, rfe
rfe(X[columnsofinterest],y)

###APP1 - TRAIN ON A FIXED SEED AND CLASSIFY WITH RF
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
# randomforestClassify(X_train,y_train,X_val,y_val,ftrain.keys())




#######################
# VIEW AND SAVE RESULTS
#######################
#sampling_results.quantile([0, 1])


# # Compute change in survival % between training and validation set
# sr['train-val-diff'] = abs(sr['rootc_train'] - sr['rootc_val'])
# print(sampling_results.head()) 
# # Plot
# fig, axs = plt.subplots(1, 1)
# axs.hist(sr['train-val-diff'], bins = np.arange(0, 0.2, step = 0.01), density = True, edgecolor = 'black', align = 'right')

# axs.set_facecolor((242/255, 242/255, 242/255))
# plt.xticks(np.arange(0.02, 0.22, step = 0.02))
# plt.yticks(np.arange(2, 24, step = 2))
# axs.xaxis.set_major_formatter(mtick.PercentFormatter(xmax = 1, decimals = 0))
# axs.yaxis.set_major_formatter(mtick.PercentFormatter(xmax = 100, decimals = 0))

# plt.text(x = 0.11, y = 20, s = "Total Data Splits: 200K", fontsize = 13)
# plt.xlabel('% Difference: Training vs. Validation data', fontsize = 13)
# plt.ylabel('% of Data Splits', fontsize = 14)
# plt.style.use(plt.style.available[11])  # 0, 11, 12, 13, 14, 15, 16
# plt.show()

# # p = sns.distplot(sr.survive_diff, norm_hist = True, bins = 19, kde = False, 
# #                  hist_kws = dict(edgecolor = "black"), label = "TEST")
# p = sns.histplot(data=sr, x="train-val-diff",bins=19)

# p.set(xlabel = '% Difference between Training and Validation data', 
#       ylabel = '% of Data Splits')
# #p.set_xticklabels(np.arange(0, 0.2, step = 0.0))
# #p.set_yticklabels(np.arange(0, 22, step = 2))
# sns.despine()
# #p.xticks(np.arange(0, 0.2, step = 0.02))
# #plt.show()
# plt.savefig("seaborn_plot.png")