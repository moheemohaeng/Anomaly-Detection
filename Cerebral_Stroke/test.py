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

#1.정상으로만 학습, 테스트는 정상과 이상 반반
answer_label = 'stroke'
X = df[df.columns.difference([answer_label])]
df_normal = df[df[answer_label] == 0]
df_abnormal = df[df[answer_label] == 1]
test_normal_df = df_normal.sample(n=420, random_state = 0)
test_df = pd.concat([df_abnormal, test_normal_df])
X_test = test_df[test_df.columns.difference([answer_label])]
y_test = test_df[answer_label]
train_df = df_normal.drop(test_normal_df.index)
X_train = train_df[train_df.columns.difference([answer_label])]
y_train = train_df[answer_label]


#2.그냥 데이터 이용
# X = df.drop('stroke',axis=1)
# y = df['stroke']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


#3.Oversampling by Smote
# overs = SMOTE(random_state=1)
# X = df.drop('stroke',axis=1)
# y = df['stroke']
# X_os, y_os = overs.fit_resample(X,y)
# X_train,X_test,y_train,y_test = train_test_split(X_os,y_os,test_size=0.3,random_state=1)


#4.Undersampling by OSS
# X = df.drop('stroke',axis=1)
# y = df['stroke']
# unders = OneSidedSelection(n_neighbors=1, n_seeds_S=1)
# X_us,y_us = unders.fit_resample(X,y)
# X_train,X_test,y_train,y_test = train_test_split(X_us,y_us,test_size=0.3,random_state=1)




# Gaussian Density Estimation
gde = EllipticEnvelope()
gde.fit(X_train)
GDE_test_pred = gde.predict(X_test) 
GDE_test_pred = pd.DataFrame(GDE_test_pred)
GDE_test_pred = GDE_test_pred.replace({-1: 1, 1: 0})

print("++++++++Gaussian Density Estimation++++++++")
print("accuracy: ", accuracy_score(y_test, GDE_test_pred))
print("recall: ", round(recall_score(y_test, GDE_test_pred),3))
print("precision: ", round(precision_score(y_test, GDE_test_pred),3))
print("f1-score: ", round(f1_score(y_test, GDE_test_pred),3))
print("===========================================")









# Mixture of Gaussian
# lowest_bic = np.infty
# bic = []
# n_components_range = range(1, 7)
# cv_types = ["spherical", "tied", "diag", "full"]
# for cv_type in cv_types:
#     for n_components in n_components_range:
#         # Fit a Gaussian mixture with EM
#         gmm = GaussianMixture(
#             n_components=n_components, covariance_type=cv_type
#         )
#         gmm.fit(X)
#         bic.append(gmm.bic(X))
#         if bic[-1] < lowest_bic:
#             lowest_bic = bic[-1]
#             best_gmm = gmm
# y_gmm = best_gmm.fit_predict(X)
# score = best_gmm.score_samples(X)
# df['score'] = score
# pct_threshold = np.percentile(score, 4)
# df['anomaly_gmm_pct'] = df['score'].apply(lambda x: 1 if x < pct_threshold else 0)
# df['anomaly_gmm_value'] = df['score'].apply(lambda x: 1 if x < pct_threshold else 0)
# KNN_test_pred = best_gmm.predict(X_test) 
# KNN_test_pred = pd.DataFrame(KNN_test_pred)
# KNN_test_pred = KNN_test_pred.replace({1:0,2:0,3:0})


# print("++++++++++++Mixture of Gaussian+++++++++++++")
# print("accuracy: ", accuracy_score(y_test, KNN_test_pred))
# print("recall: ", round(recall_score(y_test, KNN_test_pred, average = 'micro'),3))
# print("precision: ", round(precision_score(y_test, KNN_test_pred, average = 'micro'),3))
# print("f1-score: ", round(f1_score(y_test, KNN_test_pred, average = 'micro'),3))
# print("===========================================")


# con_mat = confusion_matrix(y_test, KNN_test_pred)

# print(con_mat)




# K-Nearest_Neighbors
#Hyperparameter
grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
g_res = gs.fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors = 5, weights = 'distance',algorithm = 'brute',metric = 'minkowski')
knn.fit(X_train, y_train)
KNN_test_pred = knn.predict(X_test)
print("++++++++++++K-Nearest_neighbors+++++++++++++")
print("accuracy: ", accuracy_score(y_test, KNN_test_pred))
print("recall: ", round(recall_score(y_test, KNN_test_pred),3))
print("precision: ", round(precision_score(y_test, KNN_test_pred),3))
print("f1-score: ", round(f1_score(y_test, KNN_test_pred),3))
print("===========================================")





