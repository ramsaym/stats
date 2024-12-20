
USAGE: 
Local CSV or Cloud Bucket CSV (via gsc storage python)

`python3 ./stats.py "dndc_data/biogeodb.csv" -999 predicted='Crop 1.23_RootC' threshold1=.25 threshold2=.50 "dndc" -999`
Postgre DB via Cloud SQL Instance

`python3 ./stats.py -999 agdata-378419:northamerica-northeast1:agdatastore 'Crop 1.23_RootC' .25 .50 "day_fieldcrop_1_day_fieldmanage_1" "postgres"`
