import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter
import datetime as dt


#one hot encode a categorical panda df col, categories specified by key words -> 0 
#key_words: array of string 
def onehot_encode(df):
        df_new = pd.DataFrame()
        for col in df.columns.tolist():
                s = pd.get_dummies(df[col],dtype=int)
                for s_col in s.columns.tolist():
                        df_new[col+":"+s_col] = s[s_col]
        return df_new

#encode the time to month+year 
def to_month_encode(series):
        df_new = pd.DataFrame()
        df_new['YearMonth'] = pd.to_datetime(series,format= '%Y-%m-%d' ).map(lambda x: 100*x.year + x.month)
        for t in df_new['YearMonth'].unique():
                if t<202000:
                        df_new['YearMonth'] = df_new['YearMonth'].replace(t,'202000')
        return pd.get_dummies(df_new['YearMonth'],dtype=int)


#enocde yes/no varible to 1/-1, nan to 0 
def binary_encode(df):
        df_new = pd.DataFrame()
        for col in df.columns.tolist():
                    s = pd.get_dummies(df[col],dtype=int)
                    for s_col in s.columns.tolist():
                            df_new[col+s_col] = s[s_col]
                            print('finish binary for'+col)
        return df_new

#max_min rescale numeric df col to a given range tuple
def scaling(df,rg=(-1,1)):
        df_new = pd.DataFrame();
        min_max_scaler = preprocessing.MinMaxScaler(feature_range = rg)
        for col in df.columns.tolist():
            df_new[col] = df[col].fillna(df[col].mode(dropna=True)[0]) 
            df_new[col] = min_max_scaler.fit_transform(df_new[[col]])
            print('finish maxmin scaling for '+col)
        return df_new

def hash_encode(df, num_bits):
        df_new = pd.DataFrame()
        for col,i in zip(df.columns.tolist(),range(len(df. columns))):
            encoder = ce.HashingEncoder(n_components = num_bits[i], cols = col)
            df_new = pd.concat([df_new,encoder.fit_transform(df[col])],axis = 1)
            print('finish hash encoding'+col)
        return df_new


def class_encode(df):
        df_new = pd.DataFrame()
        for col in df.columns.tolist():
                    s = pd.get_dummies(df[col],dtype=int)
                    df_new[col] = s['Yes']-s['No']
                    print('finish class for'+col)
        return df_new

def add_feature(df,path,keys):
        df2 = pd.read_csv(path)
        df = pd.merge(df, df2, how="left", on=keys)
        return df

def find_tier(tier_df,county,time):
    if county=='Unknown':
        return 'Unknown'
    if time=='Unknown':
        return 'Unknown'
 
    county_history = tier_df.loc[county,:]
    for start,end in zip(county_history.index[:-1],county_history.index[1:] ):       
        if start <= time < end:
            return str(county_history[start])
    return 'Unknown'
 
def add_count_symptoms(df,symptom_list,key):
    df['num_of_'+key] = pd.Series([0 for x in range(len(df.index))])
    for symptom in symptom_list:
        df['num_of_'+key] += (df[symptom]=='Yes').astype(int) 
    return df

        

###clean data
file_path = 'CA_combined_csv.csv'
df_raw = pd.read_csv(file_path,sep=',',error_bad_lines=False, engine="python")
df_raw = df_raw.replace({'MISSING':'Unknown','Missing':'Unknown'})
ca_counties = ["ALAMEDA", "ALPINE", "AMADOR", "BUTTE", "CALAVERAS",
                "COLUSA", "CONTRA COSTA", "DEL NORTE", "EL DORADO",
                "FRESNO", "GLENN", "HUMBOLDT", "IMPERIAL", "INYO",
                "KERN", "KINGS", "LAKE", "LASSEN", "LOS ANGELES",
                "MADERA", "MARIN", "MARIPOSA", "MENDOCINO", "MERCED",
                "MODOC", "MONO", "MONTEREY", "NAPA", "NEVADA", "ORANGE",
                "PLACER", "PLUMAS", "RIVERSIDE", "SARCRAMENTO", "SAN BENITO",
                "SAN BERNARDINO", "SAN DIEGO", "SAN FRANCISCO", "SAN JOAQUIN",
                "SAN LUIS OBISPO", "SAN MATEO", "SANTA BARBARA", "SANTA CLARA",
                "SANTA CRUZ", "SHASTA", "SIERRA", "SISKIYOU", "SOLANO", "SONOMA",
                "STANISLAUS", "SUTTER", "TEHAMA", "TRINITY", "TULARE",
                "TUOLUMNE", "VENTURA", "YOLO", "YUBA"]
df_raw.res_county.replace('BERKELEY','ALAMEDA',inplace= True)
df_raw.res_county.replace('LONG BEACH','LOS ANGELES',inplace= True)
df_raw.res_county.replace('PASADENA','LOS ANGELES',inplace= True)
df_raw =  df_raw[df_raw["res_county"].isin(ca_counties)] 
df_raw['time'] =pd.to_datetime(df_raw['cdc_case_earliest_dt'],format= '%Y-%m-%d')
df_raw.fillna('Unknown',inplace=True)
df_raw["age_raceethnicity_combined"] = df_raw["race_ethnicity_combined"] + df_raw["age_group"]
print('data cleaned')




#add features
df_add = add_feature(df_raw,'income_race_imputed.csv',['race_ethnicity_combined','res_county'])
df_add = add_feature(df_add,'demographics_df_imputed.csv','res_county')
df_add = add_feature(df_add,'geographic_region.csv','res_county')
df_add['time'] =pd.to_datetime(df_add['cdc_case_earliest_dt'],format= '%Y-%m-%d')
tier_df = pd.read_csv('ca_county_tiers_imputed.csv',index_col=0)
tier_df.columns = pd.to_datetime(tier_df.columns,format= '%Y-%m-%d')

df_add['tier'] = [find_tier(tier_df,county,time) for (county,time) in zip(df_add['res_county'],df_add['time']) ]
df_add = add_feature(df_add,'vaccine_time_county_imputed.csv',['res_county','cdc_case_earliest_dt'])
df_add['total_partially_vaccinated'].fillna(0, inplace=True)
df_add['cumulative_fully_vaccinated'].fillna(0, inplace=True)
df_add.fillna('Unknown',inplace=True)
print('feature added')

one_hot_list = ['race_ethnicity_combined','res_county','sex','age_group','age_raceethnicity_combined','geographic_region',
                'tier','death_yn','hosp_yn','icu_yn','hc_work_yn','pna_yn', 'abxchest_yn',
       'acuterespdistress_yn', 'mechvent_yn',
                'fever_yn','sfever_yn','chills_yn','myalgia_yn','runnose_yn','sthroat_yn','cough_yn','sob_yn',
                'nauseavomit_yn','headache_yn','abdom_yn','diarrhea_yn','medcond_yn']
time_list = 'cdc_case_earliest_dt'
numerical_list = ['num_of_symptoms','num_of_severe_symptoms','estimated household income', 'senior population rate',
       'teenager population rate', 'population', 'adult uninsured rate',
       'children uninsured rate','total_partially_vaccinated', 'cumulative_fully_vaccinated']
symptom_list = ['fever_yn', 'sfever_yn',
       'chills_yn', 'myalgia_yn', 'runnose_yn', 'sthroat_yn', 'cough_yn',
       'sob_yn', 'nauseavomit_yn', 'headache_yn', 'abdom_yn', 'diarrhea_yn',
       'medcond_yn']
severe_symptom_list = ['pna_yn', 'abxchest_yn',
       'acuterespdistress_yn', 'mechvent_yn']
df_add = add_count_symptoms(df_add,symptom_list,'symptoms')
df_add = add_count_symptoms(df_add,severe_symptom_list,'severe_symptoms')

df_add.to_csv(r'./CA_cleaned.csv',index = False)
df_encoded = pd.concat([onehot_encode(df_add[one_hot_list]),to_month_encode(df_add[time_list]),df_add[numerical_list]],axis = 1)
df_encoded.to_csv(r'./CA_encoded.csv',index = False)
print('finish encoding')
