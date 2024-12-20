### Summary

This repository contains code supporting a blog post on random seed best practices.   
The blog post can be found here: https://towardsdatascience.com/how-to-use-random-seeds-effectively-54a4cd855a79   
I specifically analyze how the choice of random seed can impact train/test data splits and predictive model performance.   

### Files

- "random_seed_python.py" and "random_seed_r.R": random seed testing scripts in Python and R respectively   
- CSV files: contain random seed testing results   
- "titanic" folder: contains the Titanic data which is used for random seed testing


USAGE: 
Local CSV or Cloud Bucket CSV (via gsc storage python)
`python3 ./stats.py "dndc_data/biogeodb.csv" -999 predicted='Crop 1.23_RootC' threshold1=.25 threshold2=.50 "dndc" -999`
Postgre DB via Cloud SQL Instance
`python3 ./stats.py -999 agdata-378419:northamerica-northeast1:agdatastore 'Crop 1.23_RootC' .25 .50 "day_fieldcrop_1_day_fieldmanage_1" "postgres"`
