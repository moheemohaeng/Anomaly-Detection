import pandas as pd
import numpy as np


data = pd.read_csv('mammography.csv')
print('columns : ',len(data.columns), ' -> ', end='')
correlation_matrix = data.corr()

threshold = 0.8
columns_to_remove = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            columns_to_remove.append(colname)

# 중복된 열을 제거
columns_to_remove = list(set(columns_to_remove))

# 데이터셋에서 상관계수가 높은 열 제거
data = data.drop(columns=columns_to_remove)
print(len(data.columns))

data.to_csv('modified_dataset.csv', index=False)

# 결과 데이터프레임 출력
# print(data.columns)