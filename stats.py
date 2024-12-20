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
from google.cloud.sql.connector import Connector
from google.cloud import storage
import psycopg2
import numpy as np
from scipy.stats import entropy
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
CFKEY = sys.argv[6] 
###CREDS
DB_USER = "postgres"
DB_NAME = "postgres"
DB_PASS = sys.argv[7]
#-----MAIN RUN LOGIC-----------------------------------------------------#
#-----USAGE: python3 ./createView6.py agdata-378419:northamerica-northeast1:agdatastore postgres createView-Day_SoilC_1 Day_SoilN_1 '_Day,_Crop:[0-9],[0-9]' 'x1,x2,y1,y2,_Year,Year,_Day,Day'
#---CONFIGURE DB---##############################################################################################
#following https://cloud.google.com/sql/docs/postgres/connect-instance-auth-proxy?hl=en for the cloud SQL proxy
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
engine = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=connection
)

def calculate_variance_entropy(conn, table_name, column_name):   
    conn.execute(sqlalchemy.text(f"SELECT {column_name} FROM {table_name}"))
    column_data = conn.fetchall()
    conn.close()
    column_data = [item[0] for item in column_data]
    variance = np.var(column_data)
    value_counts = np.bincount(column_data)
    ent = entropy(value_counts)
    return variance, ent


#CREATE custom function to look at variance, entropy, etc on each column and return a list of columns to pass into the final convergence join
#that we will select from to do stats.
def scanPredicateTables(tables,conn):
    tblnum=1
    collist=[]
    for tbl in tables:
        for obj in fetchHeaders(engine,tbl):
            col = obj['Column']
            variance, ent = calculate_variance_entropy(conn,tbl, col)
            print(f"Table: {tbl},Col: {col},Variance: {variance}, Entropy: {ent}")
            #qry=sqlalchemy.text(f'SELECT * FROM "{tbl}" WHERE "{col}"::text ~ \'{regex}\' limit {limit}')
            # if "interesting" is True:
            #     collist.append({f'\"{tblnum}\":\"{col}\"'})
        tblnum+=1

    return collist

#####SETUP######################################################
#################################################################
mode=-999
try:
    if INSTANCE_CONNECTION_NAME != -999:
        train_all = pd.read_sql('SELECT int_column, date_column FROM test_data', engine)
        mode=0
    else: 
        train_all = pd.read_csv(datafile)
        mode=1
    print(train_all.columns)
    #ftrain = train_all[train_all['Crop 1.23_RootC'] > 0]
    ftrain = train_all.loc[train_all['Crop 1.23_RootC'] > 0, :]
    cfg = f'{CFKEY}_stats_config.json'
    print(f"-       LOOKING FOR CONFIG FILE {cfg}")
    try:
        with open(cfg, 'r') as config_file:
            configData = json.load(config_file)
            excludeColumns = configData["columnsToExclude"]
    except: 
            configData = None
            excludeColumns = []
    print(f"--      FOUND {len(ftrain.columns.to_list())} COLUMNS TOTAL")
except Exception as e:
    print(f"!!---ERROR, MISSING PARAMS OR NO CONFIG")
    print(e)
    sys.exit(0)

print(f"---     SETTING UP - HANDLING CALL FOR {datafile} and column={COL}")
sampling_rows = 200
VERBOSE=True
SAMPLE=False
sampling_results = pd.DataFrame(np.nan, index = range(sampling_rows), columns = ['seed', 'rootc_train', 'rootc_max', 'rootc_val', 'rootc_max'])
print(f"----    SETTING UP - DROPPING {COL} FROM X DATASET")
y = ftrain['Crop 1.23_RootC'].astype('int64')
X = dropColumnList(ftrain,excludeColumns)


    
    

####PROCESS#####################################################
#################################################################
print(f"------  ANALYZING PREDICTORS OF {COL}")
if VERBOSE: 
    print(X.loc[:,X.keys().to_list()].head())

if SAMPLE:
    from rf import sampleAcrossSeeds
    print(f"SAMPLING FOR SEED SENSTIVITY: {sampling_rows} iterations")
    sampleAcrossSeeds(sampling_results,sampling_rows)

###SERVC0 - Feature Selection: TRAIN AND CLASSIFY WITH RF RETURN FEAT IMPRTNC BY QUANTILE RANK ################
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