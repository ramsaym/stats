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
from utils import *
import sqlalchemy
from sqlalchemy import text
from google.cloud import storage
import psycopg2
import numpy as np
from scipy.stats import entropy
import datetime
from pgsql import *
from rf import *
#USAGE PGSQL:   python3 ./stats.py -999 agdata-378419:northamerica-northeast1:agdatastore 'Crop 1.23_RootC' .25 .50 "day_fieldcrop_1_day_fieldmanage_1" "postgre"
#USAGE CSV:     python3 ./stats.py "dndc_data/biogeodb.csv" -999 predicted='Crop 1.23_RootC' threshold1=.25 threshold2=.50 "dndc" -999
#####PARAMS######################################################
#################################################################
name_of_script = sys.argv[0]
datafile = sys.argv[1]
INSTANCE_CONNECTION_NAME = sys.argv[2]
COL = sys.argv[3]
TH1 = sys.argv[4]
TH2 = sys.argv[5] 
DATASOURCE = sys.argv[6] 
###CREDS
DB_USER = "postgres"
DB_NAME = "postgres"
DB_PASS = sys.argv[7]
BYPASS = sys.argv[8]
SCAN = sys.argv[9]
QAREGEX = sys.argv[10]
#'_RootC_kgC/ha'
#-----USAGE: 
#target a specific column on a table and run AI regression on it after separating out Y from dataset
#python3 ./stats.py -999 agdata-378419:northamerica-northeast1:agdatastore '_RootC_kgC/ha' .25 .50 "entropy" "postgres" no no 'dd1,Day:[0-9],[0-9]' '_RootC_kgC/ha'

#following https://cloud.google.com/sql/docs/postgres/connect-instance-auth-proxy?hl=en for the cloud SQL proxy
#####-----CONFIGURE DB---###############################
connector=Connector()
# function to return the database connection object
def connection():
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME
    )
    return conn
engine = sqlalchemy.create_engine("postgresql+pg8000://",creator=connection)
###Primary Decision Fork################################
########################################################
########################################################
if INSTANCE_CONNECTION_NAME != -999:
    threshold=.1
    #MAIN ROUTE FOR PRODUCING STATS
    if (BYPASS=='no'):
        #####-----01--SQL High Entropy/Variance Column Scan
        if (SCAN=='yes'):
            interestingcolumns = scanPredicateTables([DATASOURCE],engine,threshold)
            df = pd.DataFrame(interestingcolumns)
            print(f'Columns meeting entropic threshold of: {threshold}')
            print(df.sort_values('ent',ascending=False))
            dfToCsvCloud(df,"gs://agiot/stats",VERBOSE=True)
            print("SQL Table.Col References: ")
            print(interestingcolumns['sql'])
        #####-----02--SQL Random Forest Classification APP
        else:
            targetdf = fetchTableData(engine, 'entropy', "dd1") 
        
    #####---------03--SQL View Creation Based on Target Columns 
    else:
        entropyBasedViewSQL(QAREGEX)
        exit(0) 
######################################################## 
#####-----04----------CSV Random Forest Classification APP
else: 
    targetdf = pd.read_csv(datafile)
########################################################
########################################################

#####-----Initialize X and Y Dataframes#################
########################################################
ftrain = targetdf.loc[targetdf[COL].str.contains('[0-9]'), :].astype('float64')
print(ftrain.loc[ftrain[COL > 0],:])
cfg = f'{DATASOURCE}_stats_config.json'
print(f"-       LOOKING FOR CONFIG FILE {cfg}")
try:
    with open(cfg, 'r') as config_file:
        configData = json.load(config_file)
        excludeColumns = configData["columnsToExclude"]
except: 
        configData = None
        excludeColumns = []
print(f"--      FOUND {len(ftrain.columns.to_list())} COLUMNS TOTAL")
print(f"---     SETTING UP - HANDLING CALL FOR {datafile} and column={COL}")
sampling_rows = 200
VERBOSE=True
SAMPLE=False
sampling_results = pd.DataFrame(np.nan, index = range(sampling_rows), columns = ['seed', 'rootc_train', 'rootc_max', 'rootc_val', 'rootc_max'])
print(f"----    SETTING UP - DROPPING {COL} FROM X DATASET")
#y = ftrain[COL].str.strip().fillna('', inplace=True)
y = ftrain[COL]
X = dropColumnList(ftrain,excludeColumns)
#.str.strip().astype('float64')
####PROCESS#############################################
########################################################
print(f"------  ANALYZING PREDICTORS OF {COL}")
if VERBOSE: 
    print(X.loc[:,X.keys().to_list()].head())

if SAMPLE:
    from rf import sampleAcrossSeeds
    print(f"SAMPLING FOR SEED SENSTIVITY: {sampling_rows} iterations")
    sampleAcrossSeeds(sampling_results,sampling_rows)

###SERVC0 - Feature Selection: TRAIN AND CLASSIFY WITH RF RETURN FEAT IMPRTNC BY QUANTILE RANK ################
#############################################################################################################

feats, accuracy, r2, forest_importances, std, trainingsplits = splitDataAndRunRf(X, y,TH1,COL,test_size = 0.2, random_state = 1,DEBUG=True)
X_train =  trainingsplits[0]
X_val =  trainingsplits[1]
y_train=  trainingsplits[2]
y_val = trainingsplits[3]
#attempt to reduce the amount of features. 
if (r2>.95):
    X = ftrain[feats.keys()]
    feats, accuracy, r2, forest_importances, std,trainingsplits = splitDataAndRunRf(X, y,TH1,COL,test_size = 0.2, random_state = 1,DEBUG=True)
    X_train =  trainingsplits[0]
    X_val =  trainingsplits[1]
    y_train=  trainingsplits[2]
    y_val = trainingsplits[3]

###SERVC1 - Feature Selection: permuatation based (column sets) using RF classifier
#############################################################################################################
from rf import permutationFeatureImportance
permutationFeatureImportance(X.keys().to_list(),X_train, X_val, y_train, y_val)


PLOT=True
#######################
# VIEW AND SAVE RESULTS
#######################
if PLOT:
    #fig, ax0 = plt.subplots()
    fig, ((ax0, ax1,ax2,ax3), (ax4, ax5, ax6,ax7),(ax8, ax9, ax10,ax11)) = plt.subplots(3, 4)
    #Series object
    forest_importances =  forest_importances[ forest_importances > forest_importances.quantile(float(TH2))]
    forest_importances.plot.bar(ax=ax0)
    ax0.set_title("Feature importances")
    ax0.set_ylabel("Mean decrease in impurity (MDI) ")
    print(forest_importances.dtypes)
    #fig.tight_layout()
    i=1
    for ft in feats.keys():
        X = ftrain[ft] 
        if 1 == 1:
            ax1.scatter(X, y)
            ax1.set_title(ft)
        elif i == 2:
            ax2.scatter(X, y)
            ax2.set_title(ft)
        elif i == 3:
            ax3.scatter(X, y)
            ax3.set_title(ft)
        elif i == 4:
            ax4.scatter(X, y)
            ax4.set_title(ft)
        elif i == 5:
            ax5.scatter(X, y)
            ax5.set_title(ft)
        elif i == 6:
            ax6.scatter(X, y)
            ax6.set_title(ft)
        elif i == 7:
            ax7.scatter(X, y)
            ax7.set_title(ft)
        elif i == 8:
            ax8.scatter(X, y)
            ax8.set_title(ft)
        elif i == 9:
            ax9.scatter(X, y)
            ax9.set_title(ft)
        elif i == 10:
            ax10.scatter(X, y)
            ax10.set_title(ft)
        elif i == 11:
            ax11.scatter(X, y)
            ax11.set_title(ft)
        else:
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