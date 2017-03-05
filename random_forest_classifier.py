# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 00:51:14 2017

@author: Chella Rm
"""

import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import random
import numpy as np 




df = pd.read_csv('C:\Users\Chella Rm\Documents\GitHub\Technical-\labels.csv')
true_data = []
min_maxer = MinMaxScaler()

true_arr = np.empty((0,40), float)
for idx, row in df.iterrows():
    if idx%2:
        scrip = df.ix[idx]['scrip']
        date = df.ix[idx]['Date']
        both = df.ix[idx][['scrip','Date']]
        temp_df = pd.read_csv('./data/'+scrip+'.csv')
        indux = temp_df[temp_df['Date']==date].index[0]
        a = min_maxer.fit_transform(temp_df[['Open','High','Low','Close']].ix[indux-9:indux].values.reshape(-1,1))
#       print a.shape
        if a.shape[0] != 40:
            print "what the hell",idx,a.shape
#        print normalize(temp_df[['Open','High','Low','Close']].ix[indux-11:indux].values,axis = 0).ravel()
        true_arr = np.append(true_arr,a.T,axis=0)

true_arr = np.c_[ true_arr, np.ones(true_arr.shape[0]) ] 
print true_arr.shape
true_df = pd.DataFrame(true_arr)

scrip_list = list(df['scrip'].unique())
false_arr = np.empty((0,40), float)
for i in range(0,1000):
    scrip = random.choice(scrip_list)
    temp_df = pd.read_csv('./data/'+scrip+'.csv')
    indux = random.randint(12,len(temp_df)-12)
    a = min_maxer.fit_transform(temp_df[['Open','High','Low','Close']].ix[indux-9:indux].values.reshape(-1,1))
    if a.shape[0] != 40:
            print "what the hells",i,a.shape,indux
    false_arr = np.append(false_arr,a.T,axis=0)
#    false_arr.append(min_maxer.fit_transform(temp_df[['Open','High','Low','Close']].ix[indux-9:indux].values.ravel()))
false_arr = np.c_[ false_arr, np.zeros(false_arr.shape[0]) ]
print false_arr.shape
false_df = pd.DataFrame(false_arr)

data = false_df.append(true_df,ignore_index = True)
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv('real_labelled_data.csv')


#data = pd.read_csv('C:\Users\Chella Rm\Documents\GitHub\Technical-\labelled_data.csv')
train, test = train_test_split(data, train_size = 0.8)

n_trees = 20
max_features = 6

rf_classifier = RandomForestClassifier(n_estimators=n_trees, max_features=max_features)
rf_classifier.fit(train.drop(40, axis = 1), train[40])

predicted = rf_classifier.predict(test.drop(40,axis = 1))
print f1_score(test[40],predicted, average='binary')

