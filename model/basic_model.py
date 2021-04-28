# -*- coding: utf-8 -*-
import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

kernel_log_data_path = 'memory_sample_kernel_log_round1_a_train.csv'
address_log_data_path = 'memory_sample_address_log_round1_a_train.csv'
mce_log_data_path = 'memory_sample_mce_log_round1_a_train.csv'
failure_tag_data_path = 'memory_sample_failure_tag_round1_a_train.csv'
PARENT_FOLDER = './tcdata' # data dir

# 计算每个agg_time区间的和
def etl(path, agg_time):
    import random
    data = pd.read_csv(os.path.join(PARENT_FOLDER, path))
    data['collect_time'] = pd.to_datetime(data['collect_time']).dt.ceil(agg_time)
    group_data = data.groupby(['serial_number','collect_time'],as_index=False).agg('sum')
    return group_data

# 设置聚合时间粒度
AGG_VALUE = 7
AGG_UNIT = 'min'
AGG_TIME = str(AGG_VALUE)+AGG_UNIT

# 为数据打标
failure_tag = pd.read_csv(os.path.join(PARENT_FOLDER,failure_tag_data_path))
failure_tag['failure_time']= pd.to_datetime(failure_tag['failure_time'])

# kernel 数据
group_min_kernel = etl(kernel_log_data_path, AGG_TIME)
merged_data = pd.merge(group_min_kernel,failure_tag[['serial_number','failure_time']],how='left',on=['serial_number'])
merged_data.describe()

merged_data['failure_tag']=(merged_data['failure_time'].notnull()) & ((merged_data['failure_time']
-merged_data['collect_time']).dt.seconds <= AGG_VALUE*60)
merged_data['failure_tag']= merged_data['failure_tag']+0
feature_data = merged_data.drop(['serial_number', 'collect_time','manufacturer','vendor','failure_time'], axis=1)

# 负样本下采样
sample_0 = feature_data[feature_data['failure_tag']==0].sample(frac=0.1)
sample = sample_0.append(feature_data[feature_data['failure_tag']==1])
print(sample.describe())

# split data into train and test sets
X = sample.iloc[:,:-1]
Y = sample['failure_tag']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

# fit model no training data
model = XGBClassifier(base_score=0.66)
model.fit(X_train, y_train)
	
print(model)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# precision tp / (tp + fp)
precision = precision_score(y_test, predictions)
print('Precision: %.2f%%' % (precision * 100.0))

# recall: tp / (tp + fn)
recall = recall_score(y_test, predictions)
print('Recall: %.2f%%' % (recall * 100.0))

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, predictions)
print('F1 score: %.2f%%' % (f1 * 100.0))

# 保存模型
pickle.dump(model, open("../user_data/model_data/model.dat", "wb"))