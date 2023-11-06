import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from pyod.models.iforest import IForest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
import os
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pyod.models.pca import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import GridSearchCV

import seaborn as sns
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')






df = pd.read_csv('data/preprocessed.csv')





#train, test data processing 1,2,3,4중 선택

# #1.정상으로만 학습, 테스트는 정상과 이상 반반
# answer_label = 'stroke'
# X = df[df.columns.difference([answer_label])]
# df_normal = df[df[answer_label] == 0]
# df_abnormal = df[df[answer_label] == 1]
# test_normal_df = df_normal.sample(n=420, random_state = 0)
# test_df = pd.concat([df_abnormal, test_normal_df])
# X_test = test_df[test_df.columns.difference([answer_label])]
# y_test = test_df[answer_label]
# train_df = df_normal.drop(test_normal_df.index)
# X_train = train_df[train_df.columns.difference([answer_label])]
# y_train = train_df[answer_label]


#2.그냥 데이터 이용
# X = df.drop('stroke',axis=1)
# y = df['stroke']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# 3.Oversampling by Smote
overs = SMOTE(random_state=1)
X = df.drop('stroke',axis=1)
y = df['stroke']
X_os, y_os = overs.fit_resample(X,y)
X_train,X_test,y_train,y_test = train_test_split(X_os,y_os,test_size=0.3,random_state=1)


#4.Undersampling by OSS
# X = df.drop('stroke',axis=1)
# y = df['stroke']
# unders = OneSidedSelection(n_neighbors=1, n_seeds_S=1)
# X_us,y_us = unders.fit_resample(X,y)
# X_train,X_test,y_train,y_test = train_test_split(X_us,y_us,test_size=0.3,random_state=1)







knn = KNeighborsClassifier()
nb = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

models = [knn,nb,dt,rf]

for model in models:
    print("MODEL NAME:",model)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print(f1_score(y_test,y_pred))