#---DEPENDENCIES---######################################
#---BUILT-IN
import datetime
import csv
import os
import os.path
import sys
import glob
import math
import re
from io import StringIO
import datetime as dt
#---INSTALLABLE MODULES
import pandas as pd
import numpy as np
import requests
import sqlalchemy
from google.cloud.sql.connector import Connector
import pg8000
from sqlalchemy import text
import psycopg2
import gcsfs
from google.cloud import storage
from tqdm import tqdm
#---PROJECT MODULES
from utils import *
from pgutils import *

###CREDS
DB_USER = "postgres"
DB_NAME = "postgres"
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

#---COMMON FUNCTIONS SPECIFIC TO VIEW CREATION
def sqlregexfilters(inputStr):
    last=0
    i=0
    SQLpairStr=''
    qacols = inputStr.split(":")[0]
    regexs = inputStr.split(":")[1]
    #print(regexs)
    if(len(inputStr.split(":")) != len(inputStr.split(":"))):
        print("DIFFERENT NUMBER OF QACOLS FROM REGEXS TO TEST")
        exit(0)
    for val in qacols.split(","):
        if len(regexs.split(",")) > 1:
            regex=regexs.split(",")[i]
        else:
            regex = regexs
        logicalOperator='AND'
        if (SQLpairStr==''):
            logicalOperator=''
        SQLpairStr = SQLpairStr + f' {logicalOperator} \"{val}\"::text ~ \'{regex}\'::text '   
        last+=1
        i+=1
    return SQLpairStr


def populatecolumnmappingsdict(colmaps,pydict):
    i=1 #important for modulus conditional
    for val in colmaps.split(","):
        if i==1 or i==2:
            key= 'x'
        elif i==3 or i==4:
            key= 'y'
        elif i==5 or i==6:
            key= "yyyy"
        elif i==7 or i==8:
            key= "dd"
        elif i==9 or i==10:
            key= "time"
        #every second time other than zero, we create a pairing. i must start at 1 or be checked for zero 0 % 2=0
        if i % 2 ==0 and i>0:
            #left: colMapstring is still index starting at 0 so we need to adjust for i=1..n as well as the lookback hence i-2
            pydict[key].append(colmaps.split(",")[i-2]) 
            #right
            pydict[key].append(val) 
        i+=1
    return pydict
    
def filterandrenanme(header,prefix,pydict,tblabbrev): 
    ascol=''
    spacetimeflag=-999
    if header=='x':
        ascol=f'{prefix}{pydict["x"][(int(tblabbrev[-1:])-1)]}' 
        spacetimeflag=1
    elif header=='y':
        ascol=f'{prefix}{pydict["y"][(int(tblabbrev[-1:])-1)]}'  
        spacetimeflag=2
    elif re.search('([\_]*Year{1}[\_]*)(?!\w)', header,re.IGNORECASE) is not None:
        ascol=f'{prefix}{pydict["yyyy"][(int(tblabbrev[-1:])-1)]}' 
        spacetimeflag=3
    # elif re.search('Month', header,re.IGNORECASE) is not None:
    #     ascol=f'{prefix}x_{pydict["mm"][(int(tblabbrev[-1:])-1)]}' 
    #     spacetimeflag=4
    elif re.search('((?<!\w)[\_]*Day\{1\}[\_]*)(?!\w)', header,re.IGNORECASE) is not None:
        ascol=f'{prefix}{pydict["dd"][(int(tblabbrev[-1:])-1)]}' 
        spacetimeflag=4
    elif header=='uid':
        ascol=f'{prefix}uid{(int(tblabbrev[-1:]))}' 
        spacetimeflag=55
    elif header=='timestamp':
        ascol=f'{prefix}time{(int(tblabbrev[-1:]))}' 
        spacetimeflag=6
    else:
        ascol=''
        spacetimeflag=-999
    return ascol,spacetimeflag


def sqlcolumnsequence(tableColSequence,tblabbrev,colmaps,trunk,keywords=['x','y','Year','Day'],verbose=False):
    stringout=''
    #renamemanifest = json.loads('{"x":"","y":"","yyyy":"","dd":""}')
    pydictskeleton={"x":[],"y":[],"yyyy":[],"dd":[],"time":[]}
    pydict = populatecolumnmappingsdict(colmaps,pydictskeleton)
    i=0
    for object in tableColSequence:
        if verbose:
            print(object)
        header=object['Column']
        comma1=','
        if (stringout==''):
            comma1=''
        if trunk is False:  
            prefix=' as '
        else:
            prefix=''
        #currently we are counting on a column called x and y being set, but the naming convention of time comes from the dndc reports
        #x and y on the other hand are extracted from the filename that comes straight from the GIS sampling point at the beginning of the workflow (fetchmeteo#1)
        cleanedsqlstr, spacetimeflag = filterandrenanme(header,prefix,pydict,tblabbrev)
        if trunk is False:  
            stringout =  stringout + f'{comma1}\"{tblabbrev}\"."{header}"{cleanedsqlstr}'
            #print(stringout) 
        ###TRUNK CODE, THE WRAPPING SELECT LIST OF COLUMNS as tbl[n]...tbl1,tbl2 for these two table joins
        else:
            if spacetimeflag <0:
                cleanedsqlstr=header
            else:
                if (spacetimeflag==1):
                    cleanedsqlstr=pydict["x"][(int(tblabbrev[-1:])-1)] # 1 or 2, less 1 for the zero start gives 0 or 1 to toggle table cols
                elif (spacetimeflag==2):
                    cleanedsqlstr=pydict["y"][(int(tblabbrev[-1:])-1)] 
                elif (spacetimeflag==3):
                    cleanedsqlstr=pydict["yyyy"][(int(tblabbrev[-1:])-1)] 
                elif (spacetimeflag==3):
                    cleanedsqlstr=pydict["dd"][(int(tblabbrev[-1:])-1)] 
                #double digits implies it does not participate in the join at bottom but is caught and renamed to avoid collisions
                elif (spacetimeflag==55):
                    cleanedsqlstr=f'uid{(int(tblabbrev[-1:]))}' 
                else:
                    cleanedsqlstr=pydict["time"][(int(tblabbrev[-1:])-1)] 
            stringout =  stringout + f'{comma1}\"{tblabbrev}\".\"{cleanedsqlstr}\"'            
    return stringout

#link this to other function via join names, this reads colmap. above dynamically assigns postfix to make x,y yyyy,dd unique
def createJoinColMappings(colMapStr):   
    i=1 #set it to be positional to work with modulus
    stringout=''
    root=''
    branch=''
    for val in colMapStr.split(","):
        logicalOperator=' AND'
        if (stringout==''):
            logicalOperator=''
        #i is set to 1 so it cannot be zero or indexing will throw an error on i-2 @109
        if i % 2 ==0:
            #colMapstring is still index starting at 0 so we need to adjust for i=1..n as well as the lookback hence i-2
            root = colMapStr.split(",")[i-2]
            branch = val
            stringout = stringout + f'{logicalOperator} tbl1.\"{root}\"::numeric = tbl2.\"{branch}\"::numeric'   
        i+=1
    return stringout

def gethighentropycolumns():
    return ['"day_fieldcrop_1_day_fieldmanage_1"."yyyy1"', '"day_fieldcrop_1_day_fieldmanage_1"."dd1"', '"day_fieldcrop_1_day_fieldmanage_1"."Precipitation"', '"day_fieldcrop_1_day_fieldmanage_1"."Radiation"', '"day_fieldcrop_1_day_fieldmanage_1"."PET"', '"day_fieldcrop_1_day_fieldmanage_1"."Transpiration"', '"day_fieldcrop_1_day_fieldmanage_1"."All_crops"', '"day_fieldcrop_1_day_fieldmanage_1"."All_crops.1"', '"day_fieldcrop_1_day_fieldmanage_1"."All_crops.2"', '"day_fieldcrop_1_day_fieldmanage_1"."All_crops.3"', '"day_fieldcrop_1_day_fieldmanage_1"."_CropID_#"', '"day_fieldcrop_1_day_fieldmanage_1"."_TDD_degree_C"', '"day_fieldcrop_1_day_fieldmanage_1"."_GrowthIndex_index"', '"day_fieldcrop_1_day_fieldmanage_1"."_Water_demand_mm"', '"day_fieldcrop_1_day_fieldmanage_1"."_Water_stress_index"', '"day_fieldcrop_1_day_fieldmanage_1"."_N_demand_kgN/ha/d"', '"day_fieldcrop_1_day_fieldmanage_1"."_Temp_Stress_index"', '"day_fieldcrop_1_day_fieldmanage_1"."_LAI_(LAI)"', '"day_fieldcrop_1_day_fieldmanage_1"."_N_fixation_kgN/ha/d"', '"day_fieldcrop_1_day_fieldmanage_1"."_Day_N_increase_kgN/ha/d"', '"day_fieldcrop_1_day_fieldmanage_1"."_TotalCropN_kgN/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_DailyCropGrowth_kgC/ha/d"', '"day_fieldcrop_1_day_fieldmanage_1"."_DayLeafGrowth_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_DayStemGrowth_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_DayRootGrowth_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_DayGrainGrowth_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_LeafC_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_StemC_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_RootC_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_GrainC_kgC/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_LeafN_kgN/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_StemN_kgN/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_RootN_kgN/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."_GrainN_kgN/ha"', '"day_fieldcrop_1_day_fieldmanage_1"."x1"', '"day_fieldcrop_1_day_fieldmanage_1"."uid1"', '"day_fieldcrop_1_day_fieldmanage_1"."yyyy2"', '"day_fieldcrop_1_day_fieldmanage_1"."dd2"', '"day_fieldcrop_1_day_fieldmanage_1"."x2"', '"day_fieldcrop_1_day_fieldmanage_1"."uid2"', '"day_soilc_1_day_soiln_1"."Unnamed: 0"', '"day_soilc_1_day_soiln_1"."yyyy1"', '"day_soilc_1_day_soiln_1"."dd1"', '"day_soilc_1_day_soiln_1"."Labile litter"', '"day_soilc_1_day_soiln_1"."Resistant litter"', '"day_soilc_1_day_soiln_1"."Microbe"', '"day_soilc_1_day_soiln_1"."Humads"', '"day_soilc_1_day_soiln_1"."Humus"', '"day_soilc_1_day_soiln_1"."SOC"', '"day_soilc_1_day_soiln_1"."SOC0-10cm"', '"day_soilc_1_day_soiln_1"." SOC10-20cm"', '"day_soilc_1_day_soiln_1"."SOC20-30cm"', '"day_soilc_1_day_soiln_1"."SOC30-40cm"', '"day_soilc_1_day_soiln_1"."SOC40-50cm"', '"day_soilc_1_day_soiln_1"."SOC50-60cm"', '"day_soilc_1_day_soiln_1"."SOC60-70cm"', '"day_soilc_1_day_soiln_1"."SOC70-80cm"', '"day_soilc_1_day_soiln_1"."SOC80-90cm"', '"day_soilc_1_day_soiln_1"."SOC90-100cm"', '"day_soilc_1_day_soiln_1"."SOC100-110cm"', '"day_soilc_1_day_soiln_1"." SOC110-120cm"', '"day_soilc_1_day_soiln_1"."SOC120-130cm"', '"day_soilc_1_day_soiln_1"."SOC130-140cm"', '"day_soilc_1_day_soiln_1"."SOC140-150cm"', '"day_soilc_1_day_soiln_1"."SOC150-160cm"', '"day_soilc_1_day_soiln_1"."SOC160-170cm"', '"day_soilc_1_day_soiln_1"."SOC170-180cm"', '"day_soilc_1_day_soiln_1"."SOC180-190cm"', '"day_soilc_1_day_soiln_1"."SOC190-200cm"', '"day_soilc_1_day_soiln_1"."DOC"', '"day_soilc_1_day_soiln_1"."DOC_produce"', '"day_soilc_1_day_soiln_1"."DOC_consume"', '"day_soilc_1_day_soiln_1"."DOC_leach"', '"day_soilc_1_day_soiln_1"."Leaf-respiration"', '"day_soilc_1_day_soiln_1"." Root-respiration"', '"day_soilc_1_day_soiln_1"."Soil-heterotrophic-respiration"', '"day_soilc_1_day_soiln_1"." NPP"', '"day_soilc_1_day_soiln_1"."Stub"', '"day_soilc_1_day_soiln_1"."DOC_from_root"', '"day_soilc_1_day_soiln_1"."Litter-C"', '"day_soilc_1_day_soiln_1"." SoilCO2_0-10cm"', '"day_soilc_1_day_soiln_1"."x1"', '"day_soilc_1_day_soiln_1"."uid1"', '"day_soilc_1_day_soiln_1"."yyyy2"', '"day_soilc_1_day_soiln_1"."dd2"', '"day_soilc_1_day_soiln_1"."_Ice_DOC"', '"day_soilc_1_day_soiln_1"."_N_fixation"', '"day_soilc_1_day_soiln_1"."_SON0-10cm"', '"day_soilc_1_day_soiln_1"."_ SON10-20cm"', '"day_soilc_1_day_soiln_1"."_SON20-30cm"', '"day_soilc_1_day_soiln_1"."_SON30-40cm"', '"day_soilc_1_day_soiln_1"."_SON40-50cm"', '"day_soilc_1_day_soiln_1"."x2"', '"day_soilc_1_day_soiln_1"."uid2"', '"day_soilclimate_1_day_soilmicrobe_1"."_0"', '"day_soilclimate_1_day_soilmicrobe_1"."yyyy1"', '"day_soilclimate_1_day_soilmicrobe_1"."dd1"', '"day_soilclimate_1_day_soilmicrobe_1"."_Prec.(mm)"', '"day_soilclimate_1_day_soilmicrobe_1"."_PET"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_50cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_60cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_70cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_80cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_90cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_100cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_110cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_120cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_130cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_140cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_150cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_160cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_170cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_180cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_190cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_temperature_200cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_moisture(wfps)"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_moisture(wfps)_5cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_moisture(wfps)_10cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_moisture(wfps)_20cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_moisture(wfps)_30cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_Eh(mv)"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_Eh(mv)_10cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_Eh(mv)_20cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_Eh(mv)_30cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_Eh(mv)_40cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_Eh(mv)_50cm"', '"day_soilclimate_1_day_soilmicrobe_1"."Ice"', '"day_soilclimate_1_day_soilmicrobe_1"."Snowpack"', '"day_soilclimate_1_day_soilmicrobe_1"."SoilWater"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_pH"', '"day_soilclimate_1_day_soilmicrobe_1"."Soil_pH_20.0"', '"day_soilclimate_1_day_soilmicrobe_1"."x1"', '"day_soilclimate_1_day_soilmicrobe_1"."uid1"', '"day_soilclimate_1_day_soilmicrobe_1"."Unnamed: 0"', '"day_soilclimate_1_day_soilmicrobe_1"."yyyy2"', '"day_soilclimate_1_day_soilmicrobe_1"."dd2"', '"day_soilclimate_1_day_soilmicrobe_1"."Decomposition (kgC/ha)"', '"day_soilclimate_1_day_soilmicrobe_1"."x2"', '"day_soilclimate_1_day_soilmicrobe_1"."uid2"', '"Day_CO2_1"."Unnamed: 0"', '"Day_CO2_1"."Year"', '"Day_CO2_1"."Day"', '"Day_CO2_1"."Precip"', '"Day_CO2_1"."cropType1"', '"Day_CO2_1"."wreq1"', '"Day_CO2_1"."maxmn1"', '"Day_CO2_1"."graincn1"', '"Day_CO2_1"."x"', '"Day_CO2_1"."uid"']

def joinsql(sqlstruct,col):
    sqlj=''
    _and=" AND "
    i=1
    for t in sqlstruct["table"]:
        #if we have 3 tables we need t1.x1=t2.x2 AND t2.x2 = t3.x3 (circe back on this assumption)
        #this prevents the num +1 table reference from overshooting
        if i< len(sqlstruct["table"]):
            if sqlj=='':
                _and=''
            sqlj = sqlj + f'{_and}tbl{i}.{col}{i}::numeric = tbl{i+1}.{col}{i+1}::numeric'
    return sqlj


def dataframefromdict(dictFieldNamesTableNum):
    i=0
    #print(dictFieldNamesTableNum)
    datastore=[]
    for sqlchunk in dictFieldNamesTableNum['cols']:
        #SQL for each table is passed as one fstring to preserve quotes on tables. we split it by comma to get each col and tnum pair
        tableCols = sqlchunk.split(",")
        for tcol in tableCols:
            #print(f'---Appending {tcol} to dataframe')
            tnum = dictFieldNamesTableNum['tnum'][i]
            datastore.append([tcol,tnum])
        i+=1
    
    #print(datastore)
    dataframe=pd.DataFrame(datastore,columns=['col','tnum'])
    #dataframe=pd.DataFrame(datastore,columns=cols)

    return dataframe


def subquery(sqldict):
    #print(sqlmapdict)
    sqldictout = {"subquery":[],"condition":[]}
    sqldf = dataframefromdict(sqldict)
    #sqldf = pd.DataFrame(sqldict, columns=['cols','tnum'])
    print(sqldf)
    sql1=''
    j=1
    #unique tables n = 1-10 on average
    for t in sqldict['table']:
        tnum=j
        cols = ''
        omissionqueue=[]
        #cols are on the order of tables * cols
        grouped = sqldf.groupby('tnum')
        for name, group in grouped:
            #print(grouped)
        #for i in range(len(sqlstruct['cols'])):
            #col = sqlstruct['cols'][i]
            #tnumcol = sqlstruct['tnum'][i]
            tnumloop=name
            col=group['col']
            if(tnumloop == tnum and col not in omissionqueue):
                comma=','
                if cols=='':
                    comma=''
                cols = cols + f'{comma}{col}'
                omissionqueue.append(col)
                #print(omissionqueue)
        sql1 = sql1 + f'(SELECT {cols} FROM {t}) tbl{tnum}'
        sqldictout['subquery'].append(sql1)
        j+=1
        
    return sqldictout


#---COMMON FUNCTIONS SPECIFIC TO VIEW CREATION
def sqlregexfilters(inputStr):
    last=0
    i=0
    SQLpairStr=''
    #e.g. 'Day,Year:[0-9],[0-9]'
    qacols = inputStr.split(":")[0]
    regexs = inputStr.split(":")[1]
    #print(regexs)
    if(len(inputStr.split(":")) != len(inputStr.split(":"))):
        print("DIFFERENT NUMBER OF QACOLS FROM REGEXS TO TEST")
        exit(0)
    for val in qacols.split(","):
        if len(regexs.split(",")) > 1:
            regex=regexs.split(",")[i]
        else:
            regex = regexs
        logicalOperator='AND'
        if (SQLpairStr==''):
            logicalOperator=''
        SQLpairStr = SQLpairStr + f' {logicalOperator} \"{val}\"::text ~ \'{regex}\'::text '   
        last+=1
        i+=1
    return SQLpairStr


def packagetablestojoin():
    #this is to mimic piping in results from the stats.py discovery component of code. can be direct or api structured
    hientcols = gethighentropycolumns()
    tables={"table":[],"col":[]}
    i=0
    for obj in hientcols:
        bits = obj.split('.')
        table = bits[0]
        tables['table'].append(table)
        col = bits[1]
        tables['col'].append(col)
        i +=1
    return pd.DataFrame(tables)


def entropyBasedViewSQL(QAREGEX,DEBUG=False):
    tblsdf = packagetablestojoin()
    #print(tblsdf)
    sqldict = {"cols":[],"colstrunk":[],"table":[],"tnum":[]}
    joinmap = {"x":[],"y":[],"dd":[],"yyyy":[]}
    i=1
    #strip as much meta data as possible
    for t in tblsdf['table'].unique():
        #tbl1col sequence
        sql=''
        sqltrunk = ''
        j=0
        for c in tblsdf['col']:
            #matching line here to only take tables from the DF if they macth the outerloop table
            if (tblsdf['table'][j] == t):
                
                xmatch = re.search("[\_x]*[0-9]{0,1}$",c)
                if xmatch is not None:
                    joinmap['x'].append(f'x{i}')
                ymatch = re.search("[\_y]*[0-9]{0,1}$",c)
                if ymatch is not None:
                    joinmap['y'].append(f'y{i}')
                yearmatch = re.search("[\_Year,year,yyyy]*[0-9]{0,1}$",c)
                if yearmatch is not None:
                    joinmap['yyyy'].append(f'yyyy{i}')
                daymatch = re.search("[\_Day,day,dd]*[0-9]{0,1}$",c)
                if daymatch is not None:
                    joinmap['dd'].append(f'dd{i}')
                
                #format sql string
                comma=','
                if(sql==''):
                    comma=''
                sql = sql +  f'{comma}{t}.{c}'
                sqltrunk = sqltrunk + f'{comma}tbl{i}.{c}'
                sqldict['tnum'].append(i)
                #print(tblsdf['table'][j] + "= " + t + "tnum: " + str(sqldict['tnum'][j]))
            j+=1
        print("analyzing " + t)
        sqldict['cols'].append(sql)
        sqldict['colstrunk'].append(sqltrunk)
        sqldict['table'].append(t)
        
        i +=1
    trunkcols = sqldict['colstrunk']
    if DEBUG:
        print(sqldict)
    qryRaw = f'CREATE MATERIALIZED VIEW public.entropy TABLESPACE pg_default AS SELECT {trunkcols} FROM'
    #sqlview['trunk'] = f'(SELECT {trunkcols} FROM'
    subq = subquery(sqldict)
    #now we have a collection of ready to go selects    
    for sv in subq['subquery']:
        s=''
        if (s==''):
            qryRaw = qryRaw + sv
        else:
            s=" JOIN " 
        qryRaw = qryRaw + s
    ##create final join str
    qryRaw = qryRaw + " ON " 
    # tbl1.x1::numeric = tbl2.x2::numeric AND tbl1.y1::numeric = tbl2.y2::numeric AND tbl1.yyyy1::numeric = tbl2.yyyy2::numeric AND tbl1.dd1::numeric = tbl2.dd2::numeric
    xsqlj = joinsql(sqldict,"x")
    ysqlj = joinsql(sqldict,"y")
    yrsqlj = joinsql(sqldict,"yyyy")
    dsqlj = joinsql(sqldict,"dd")
    qryRaw = qryRaw + f'{xsqlj} AND {ysqlj} AND {yrsqlj} AND {dsqlj}'
    qry = sqlalchemy.text(qryRaw)
    print(qry)
