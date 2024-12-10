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

datafile = sys.argv[1]
try:
    FOCUS = sys.argv[2]
    COL = sys.argv[3]
    TH1 = sys.argv[4]
    TH2 = sys.argv[5] 
    CFKEY = sys.argv[6] 
    train_all = pd.read_csv(datafile)
    ftrain = train_all[train_all['Crop 1.23_RootC'] > 0]
    cfg = f'{CFKEY}_stats_config.json'
    print(f"-       LOOKING FOR CONFIG FILE {cfg}")
    with open(cfg, 'r') as config_file:
        configData = json.load(config_file)
    columnsofinterest  = configData["columnsofinterest"]
    excludeColumns = configData["columnsToExclude"]
    print(f"--      FOUND {len(columnsofinterest)} COLUMNS OF INTEREST")
    print(f"--      FOUND {len(ftrain.columns.to_list())} COLUMNS TOTAL")
except:
    print(f"!!---ERROR, MISSING PARAMS OR NO CONFIG: {cfg}")
    sys.exit(0)


#####SETUP######################################################
#################################################################
print(f"---     SETTING UP - HANDLING CALL FOR {datafile} focus={FOCUS} and column={COL}")
sampling_rows = 200
VERBOSE=True
SAMPLE=False
sampling_results = pd.DataFrame(np.nan, index = range(sampling_rows), columns = ['seed', 'rootc_train', 'rootc_max', 'rootc_val', 'rootc_max'])
print(f"----    SETTING UP - DROPPING {COL} FROM X DATASET")
y = ftrain['Crop 1.23_RootC'].astype('int64')
if FOCUS =='y':  
    print("!--- FOCUS IS SET") 
    X = ftrain[columnsofinterest].drop(columns=excludeColumns) 
else:   
    X = ftrain.drop(columns=excludeColumns) # Split up dependent and independent variables
    

####PROCESS#####################################################
#################################################################
print(f"------  ANALYZING PREDICTORS OF {COL}")
if VERBOSE: 
    print(ftrain.loc[:,X.keys().to_list()].head())

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
#print(feats.keys())
print(feats)
print(f"1-R^2:{r2}")
###########RUN 1##################
if (r2>.95):
    X = ftrain[feats.keys()]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)
    print(f"------- FEATURE IMPORTANCE USING {TH2} qUANTILE THRESHOLD")
    feats, accuracy, r2, forest_importances, std = randomforestAnalyze(X_train,y_train,X_val,y_val,feats.keys(),identifier=COL,thresholdQuant=TH2)
    print(f"2-R^2:{r2}")
    print(feats)



from rf import permutationFeatureImportance
permutationFeatureImportance(X.keys().to_list(),X_train, X_val, y_train, y_val)

PLOT=True
#######################
# VIEW AND SAVE RESULTS
#######################
if PLOT:
    #fig, ax0 = plt.subplots()
    fig, ((ax0, ax1,ax2,ax3), (ax4, ax5, ax6,ax7),(ax8, ax9, ax10,ax11)) = plt.subplots(3, 4)
    forest_importances.plot.bar(yerr=std, ax=ax0)
    ax0.set_title("Feature importances")
    ax0.set_ylabel("Mean decrease in impurity (MDI) ")
    fig.tight_layout()
    i=1
    for ft in feats.keys():
        X = ftrain[ft] 
        #print(X)
        #Use 5 remaining slots to fill with top four features. Returned as feats sorted by MDI from rf.py
        match i:
            case 1:
                ax1.scatter(X, y)
                ax1.set_title(ft)
            case 2:
                ax2.scatter(X, y)
                ax2.set_title(ft)
            case 3:
                ax3.scatter(X, y)
                ax3.set_title(ft)
            case 4:
                ax4.scatter(X, y)
                ax4.set_title(ft)
            case 5:
                ax5.scatter(X, y)
                ax5.set_title(ft)
            case 6:
                ax6.scatter(X, y)
                ax6.set_title(ft)
            case 7:
                ax7.scatter(X, y)
                ax7.set_title(ft)
            case 8:
                ax8.scatter(X, y)
                ax8.set_title(ft)
            case 9:
                ax9.scatter(X, y)
                ax9.set_title(ft)
            case 10:
                ax10.scatter(X, y)
                ax10.set_title(ft)
            case 11:
                ax11.scatter(X, y)
                ax11.set_title(ft)
            case _:
                print("!--- Unmapped Plot/Out of grid slots.")
        
        i+=1
    
    plt.show()
    plt.savefig(f"panel_featImportance_{COL}.png")

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