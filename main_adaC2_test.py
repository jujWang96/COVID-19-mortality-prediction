import pandas as pd
import numpy as np
from numpy import random
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score,recall_score
from sklearn.datasets import make_classification
from sklearn import model_selection, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import initializers
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import AdaC2
import pickle
def readDat_ICU(data_file_path,size = "full"):
        if size=="full":
                df = pd.read_csv(data_file_path, sep=',')
        else:
                df = pd.read_csv(data_file_path, sep=',', nrows = size)
        drop_column = ["estimated household income","senior population rate","teenager population rate",
               "population","adult uninsured rate",
               "children uninsured rate",
               "total_partially_vaccinated",
               "cumulative_fully_vaccinated"]
        target_column = ["death_yn:No",
                        "death_yn:Unknown",
                        "death_yn:Yes",
                        "icu_yn:No",
                        "icu_yn:Unknown",
                        "icu_yn:Yes"]
        truncate_df = df[~((df["death_yn:Unknown"]==1) & (df["icu_yn:Unknown"]==1))]
        features = truncate_df.drop(columns = target_column)
        #features = features.drop(columns = drop_column)

        target = (truncate_df["death_yn:Yes"]==1) | (truncate_df["icu_yn:Yes"]==1)
        return features,target

def readDat(data_file_path,size = "full"):
        if size=="full":
                df = pd.read_csv(data_file_path, sep=',')
        else:
                df = pd.read_csv(data_file_path, sep=',', nrows = size)
        drop_column = ["estimated household income","senior population rate","teenager population rate",
               "population","adult uninsured rate",
               "children uninsured rate",
               "total_partially_vaccinated",
               "cumulative_fully_vaccinated"]
        
        
        target_column = ["death_yn:No",
                         "death_yn:Unknown",
                         "death_yn:Yes"]
        truncate_df = df[~(df["death_yn:Unknown"]==1)]
        features = truncate_df.drop(columns = target_column)
       # features = features.drop(columns = drop_column)

        target = (truncate_df["death_yn:Yes"]==1)

        return features,target 
       

def main():
        data_file_path = 'data/CA_encoded.csv'
        X,y = readDat(data_file_path)
        input_size = X.shape[0]
        test_prop = 0.15

        #X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=1)
        X_test = X.iloc[(round(input_size*(1-test_prop))+1):,:]
                          
        y_test = y.iloc[round(input_size*(1-test_prop))+1:,] 
        fw =0.484
        file_to_read = open("adac2_fw"+str(fw)+".pickle", "rb")
        selected_clf = pickle.load(file_to_read)
        print('loaded')
        file_to_read.close()
        print('trainning')
        y_predtest= selected_clf.predict(X_test.to_numpy())

        print('metric of prediction on the last {} percentage data is accuracy: {}, recall: {}'.format(test_prop*100,\
                                                                                                       accuracy_score(y_test.to_numpy(),y_predtest),\
                                                                                                       recall_score(y_test.to_numpy(),y_predtest)) )              
                        

       
if __name__=="__main__":
        main()



