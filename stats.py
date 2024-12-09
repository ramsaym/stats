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
import json


#CALL:
#python3 ./stats.py "dndc_data/biogeodb.csv" True 'Crop 1.23_RootC' .02 .04

#####PARAMS######################################################
#################################################################
name_of_script = sys.argv[0]
try: 
    datafile = sys.argv[1]
    train_all = pd.read_csv(datafile)
except:
    print("!! NO DATAFILE SPECIFIED ---")
    exit(0)
try:
    FOCUS = sys.argv[2]
except:
    FOCUS = False
try:
    COL = sys.argv[3]
except:
    COL = False
try:
    TH1 = sys.argv[4]
    TH2 = sys.argv[5] 
except:
    TH1 = False
    TH2 = False 
try:
    CFKEY = sys.argv[6]
    cfg = f'{CFKEY}_stats_config.json'
    print(f"-       LOOKING FOR CONFIG FILE {cfg}")
    with open(cfg, 'r') as config_file:
        configData = json.load(config_file)
    columnsofinterest  = configData["columnsofinterest"]
    print(f"--      FOUND {len(columnsofinterest)} COLUMNS")
except:
    CFKEY = False 
    columnsofinterest = False

#####SETUP######################################################
#################################################################
print(f"---     SETTING UP - HANDLING CALL FOR {datafile} focus={FOCUS} and column={COL}")
sampling_rows = 200
VERBOSE=True
SAMPLE=False
sampling_results = pd.DataFrame(np.nan, index = range(sampling_rows), columns = ['seed', 'rootc_train', 'rootc_max', 'rootc_val', 'rootc_max'])
ftrain = train_all[train_all['Crop 1.23_RootC'] > 0]
#It makes sense to reduce this after a few runs


print(f"----    SETTING UP - DROPPING {COL} FROM X DATASET")
y = ftrain['Crop 1.23_RootC'].astype('int64')
print(f"-----   FOCUS:{FOCUS}, COI: {columnsofinterest} ")
if FOCUS:
    if columnsofinterest is not False:
        #This takes on the columns specified by a list/array. Can be config ported. Split up dependent and independent variables
        X = ftrain[columnsofinterest].drop('Crop 1.23_RootC', axis = 1) 
    
else:
    #this takes all columns if FOCUS is false. Can be config ported.  #Split up dependent and independent variables
    X = ftrain.drop(['Crop 1.23_RootC','x1','y1'], axis = 1) # Split up dependent and independent variables
    



####PROCESS#####################################################
#################################################################
print(f"------  ANALYZING PREDICTORS OF {COL}")
if VERBOSE: 
    if columnsofinterest: print(ftrain.loc[:,columnsofinterest].head())

if SAMPLE:
    from rf import sampleAcrossSeeds
    print(f"SAMPLING FOR SEED SENSTIVITY: {sampling_rows} iterations")
    sampleAcrossSeeds(sampling_results,sampling_rows)

###APP0 - Feature Selection: TRAIN AND CLASSIFY WITH RF RETURN FEAT IMPRTNC BY QUANTILE RANK ################
#############################################################################################################
from rf import randomforestAnalyze
###APP1 - TRAIN ON A FIXED SEED AND CLASSIFY WITH RF
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(f"------- FEATURE IMPORTANCE USING {TH1} qUANTILE THRESHOLD")
##Setup to iteratate on. ~30seconds per run on large datasets
###########RUN 0##################
feats, accuracy, r2, forest_importances, std = randomforestAnalyze(X_train,y_train,X_val,y_val,X.keys(),identifier=COL,thresholdQuant=TH1)
print(feats.keys())
print(f"1-R^2:{r2}")
###########RUN 1##################
if (r2>.95):
    X = ftrain[feats.keys()]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
    print(f"------- FEATURE IMPORTANCE USING {TH2} qUANTILE THRESHOLD")
    feats, accuracy, r2, forest_importances, std = randomforestAnalyze(X_train,y_train,X_val,y_val,feats.keys(),identifier=COL,thresholdQuant=TH2)
    print(feats.keys())
    print(f"2-R^2:{r2}")



PLOT=True
#######################
# VIEW AND SAVE RESULTS
#######################
if PLOT:
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using mean decrease in impurity (MDI)")
    ax.set_ylabel("Mean decrease in impurity (MDI) ")
    sizes = np.random.uniform(15, 80, len(X))
    colors = np.random.uniform(15, 80, len(X))
    fig.tight_layout()
    ax[0, 0].plot(X,y)
    #plt.show()
    #plt.savefig(f"featImportance_{identifier}.png")
    i=0
    for ft in feats.keys():
        X = ftrain[ft]   
        ax.scatter(X, y, s=sizes, c=colors, vmin=0, vmax=100)
        ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
            ylim=(0, 8), yticks=np.arange(1, 8))
        #plt.show()
        if i%2=0:
            ax[0, i].plot(X,y)
        else:
            ax[i, 0].plot(X,y)
        i+=1

    plt.show()
    plt.savefig(f"featImportance_{COL}.png")

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