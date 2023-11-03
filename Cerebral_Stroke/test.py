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

import seaborn as sns
sns.set_style('white')

import warnings
warnings.filterwarnings('ignore')


#일반 데이터
df = pd.read_csv('data/preprocessed.csv')
X = df.drop('stroke',axis=1)
y = df['stroke']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

#Oversampling by Smote
overs = SMOTE(random_state=1)
X_os, y_os = overs.fit_resample(X,y)
X_os_train,X_os_test,y_os_train,y_os_test = train_test_split(X_os,y_os,test_size=0.3,random_state=1)

#Undersampling by OSS
unders = OneSidedSelection(n_neighbors=1, n_seeds_S=1)
X_us,y_us = unders.fit_resample(X,y)
X_us_train,X_us_test,y_us_train,y_us_test = train_test_split(X_us,y_us,test_size=0.3,random_state=1)


