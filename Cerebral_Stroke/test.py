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

import os
for dirname, _, filenames in os.walk('work/Anomaly-Detection/Cerebral_Stroke/data') :
    for filename in filename:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('data/dataset.csv')
print(df.head(3))