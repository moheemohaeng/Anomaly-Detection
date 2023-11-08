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

import os, sys, pickle
import argparse, sys

import seaborn as sns
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')





parser = argparse.ArgumentParser()
parser.add_argument('-label', help=' : Please set the label') 
args = parser.parse_args()


df = pd.read_csv('data/modified_dataset.csv')


df_stroke_0 = df[df['Cancer'] == 0]
df_stroke_1 = df[df['Cancer'] == 1]





#train, test data processing 1,2,3,4중 선택

#1.정상으로만 학습, 테스트는 정상과 이상 반반
answer_label = args.label
X = df[df.columns.difference([answer_label])]
df_normal = df[df[answer_label] == 0]
df_abnormal = df[df[answer_label] == 1]
test_normal_df = df_normal.sample(n=len(df_stroke_1), random_state = 0)
test_df = pd.concat([df_abnormal, test_normal_df])
X_test = test_df[test_df.columns.difference([answer_label])]
y_test = test_df[answer_label]
train_df = df_normal.drop(test_normal_df.index)
X_train = train_df[train_df.columns.difference([answer_label])]
y_train = train_df[answer_label]





from pyod.models.hbos import HBOS
from pyod.utils.data import generate_data

model = HBOS(contamination=0.15)
model.fit(X_train)

# 이상치 점수 계산
y_scores = model.decision_function(X_test)

# 이상치 예측
y_pred = model.predict(X_test)

print("++++++++++++++++++++HBOS+++++++++++++++++++")
print("accuracy: ", accuracy_score(y_test, y_pred))
print("recall: ", round(recall_score(y_test, y_pred),3))
print("precision: ", round(precision_score(y_test, y_pred),3))
print("f1-score: ", round(f1_score(y_test, y_pred),3))
print("===========================================")






from sklearn.svm import OneClassSVM
OCSVM = OneClassSVM(kernel='sigmoid', nu = 0.35, verbose = True)
OCSVM.fit(X_train)

OCSVM_train_pred = OCSVM.predict(X_train)
OCSVM_test_pred = OCSVM.predict(X_test)
OCSVM_test_pred = pd.DataFrame(OCSVM_test_pred)
OCSVM_test_pred = OCSVM_test_pred.replace({-1:1, 1:0})

print("+++++++++++++++One Class SVM+++++++++++++++")
print("accuracy: ", accuracy_score(y_test, OCSVM_test_pred))
print("recall: ", round(recall_score(y_test, OCSVM_test_pred),3))
print("precision: ", round(precision_score(y_test, OCSVM_test_pred),3))
print("f1-score: ", round(f1_score(y_test, OCSVM_test_pred),3))
print("===========================================")


