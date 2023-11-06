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
import tensorflow as tf
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



input_dim = X_train.shape[1]

AE = tf.keras.models.Sequential([
    
    # encode
    tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim, )), 
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(2, activation='elu'),
    
    # decode
    tf.keras.layers.Dense(4, activation='elu'),
    tf.keras.layers.Dense(input_dim, activation='elu')
    
])

AE.compile(optimizer="adam", loss="mse")

history = AE.fit(
    X_train,
    X_train,
    epochs=100,
    batch_size=32,
    validation_split=0.3
)

# train data에 대한 예측 값
AE_train_pred = AE.predict(X_train)

# 실제 값과 예측 값 사이의 차이인 MSE값을 reconstruction error로 정의 -> Novelty Score
train_mse = np.mean(np.power(X_train - AE_train_pred, 2), axis=1)
train_mse = pd.DataFrame({'Reconstruction_error': train_mse})


AE_thresh = np.percentile(sorted(train_mse['Reconstruction_error']), 90)

# test data에 대한 예측 값
AE_test_pred = AE.predict(X_test)

# 실제 값과 예측 값 사이의 차이인 MSE값을 reconstruction error로 정의
test_mse = np.mean(np.power(X_test - AE_test_pred, 2), axis=1)


# classifier
# AE_thresh 기준, test_mse 값이 더 크면 이상(1), 작으면 정상(0)으로 분류
AE_test_df = []

def novelty_classifier(novelty_score):
    for i in range(len(novelty_score)):
        if novelty_score[i] > AE_thresh:
            AE_test_df.append(1)
        else:
            AE_test_df.append(0)


# Confusion matrix 출력 및 모델 성능 평가
con_mat = confusion_matrix(y_test, AE_test_df) #confusion_matrix 함수 실행

sns.heatmap(pd.DataFrame(con_mat, columns = ['Predicted', 'Actual']),
            xticklabels=['Normal [0]', 'Abnormal [1]'], 
            yticklabels=['Normal [0]', 'Abnormal [1]'], 
            annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
plt.ylabel('Actual')
plt.xlabel('Predicted')

print("accuracy: ", accuracy_score(y_test, AE_test_df))
print("recall: ", round(recall_score(y_test, AE_test_df),3))
print("precision: ", round(precision_score(y_test, AE_test_df),3))
print("f1-score: ", round(f1_score(y_test, AE_test_df),3))