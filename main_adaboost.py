import pandas as pd
import numpy as np
from numpy import random
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
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
#import AdaBoost  
def readDat_ICUU(data_file_path,size = "full"):
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
        features = features.drop(columns = drop_column)

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
        features = features.drop(columns = drop_column)

        target = (truncate_df["death_yn:Yes"]==1) 
        return features,target

	
def class_w(arr):
    """
     calculate different class weights for trainning unbalanced data
    """
    pos = sum(arr==1)
    neg = sum(arr==0)
    total = pos+neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight

def init_bias(arr):
    """
    initialize bias term for imbalanced dataset 
    """
    pos = sum(arr==1)
    neg = sum(arr==0)
    b = np.log([pos/neg])


    return b[0]
def keras_NN_classification():

        model = Sequential()
        model.add(Dense(100, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='sigmoid',
                        bias_initializer = initializers.Constant(initial_bias)))
        #model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

        
       

def main():
        data_file_path = '../data/CA_encoded.csv'
        X,y = readDat(data_file_path,300000)
        input_dim = X.shape[1]
        X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=1)
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler.transform(X_train)
        scaler.transform(X_test)
        class_weight = class_w(y_train)
        initial_bias = init_bias(y_train)
        early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True)
    
        boosted_ann = AdaBoostClassifier(n_estimators=30, random_state=0)
        history = boosted_ann.fit(X_train, y_train)
        y_pred = boosted_ann.predict(X_test)
        
        print(f1_score(y_test,y_pred))
        print(confusion_matrix(y_test,y_pred))

       
if __name__=="__main__":
        main()



